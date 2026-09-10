use super::common::{IsolatedWorkspace, RecUi};
use super::*;
use hi_ai::test_support::{FakeOpenAiServer, Response, sse_text};
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

#[derive(Clone, Copy)]
enum TerminalProviderFailure {
    Protocol,
    Capacity,
    Outage,
}

impl TerminalProviderFailure {
    fn leftover(self) -> bool {
        !matches!(self, Self::Outage)
    }
}

#[tokio::test]
async fn physical_provider_exhaustion_after_tool_work_settles_actual_verification_once() {
    assert_provider_failure_settlement(TerminalProviderFailure::Protocol).await;
}

#[tokio::test]
async fn physical_capacity_exhaustion_after_tool_work_settles_as_leftover() {
    assert_provider_failure_settlement(TerminalProviderFailure::Capacity).await;
}

#[tokio::test]
async fn terminal_retryable_outage_after_tool_work_settles_without_extra_requests() {
    assert_provider_failure_settlement(TerminalProviderFailure::Outage).await;
}

async fn assert_provider_failure_settlement(kind: TerminalProviderFailure) {
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
        let leftover = kind.leftover();
        let expected_failed_sends = if leftover { 4 } else { 2 };
        responses.extend((0..expected_failed_sends).map(|_| match kind {
            TerminalProviderFailure::Protocol => Response::json(
                400,
                r#"{"error":{"message":"invalid tool JSON","code":"tool_protocol_error","retryable":true}}"#,
            ),
            TerminalProviderFailure::Capacity => Response::json(
                429,
                r#"{"error":{"message":"capacity temporarily unavailable","code":"capacity_unavailable","retryable":true,"retry_after_seconds":0}}"#,
            ),
            TerminalProviderFailure::Outage => Response::json(
                503,
                r#"{"error":{"message":"upstream unavailable","code":"service_unavailable","retryable":true}}"#,
            ),
        }));
        let Some(server) = FakeOpenAiServer::new(responses) else {
            return;
        };
        let mut cfg = workspace.config();
        cfg.loop_limits.max_recovery_interventions = 16;
        // Protocol retries must not request a 5th HTTP send. keep-working
        // is covered by the review-and-fix protocol-storm test.
        cfg.loop_limits.max_keep_working = if leftover { 0 } else { 2 };
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
        let (status, stop_reason, verification) = if leftover {
            let outcome = subject
                .run_turn("build all of the requested implementation", &mut ui)
                .await
                .expect("protocol and request-limit budgets are not a provider outage");
            (outcome.status, outcome.stop_reason, outcome.verification)
        } else {
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
            let provider_error = hi_ai::provider_error_details(&error)
                .expect("original provider cause survives settlement");
            assert_eq!(provider_error.kind, hi_ai::ProviderErrorKind::Outage);
            assert!(provider_error.request_failure.is_none());
            assert_eq!(
                format!("{error:#}").matches("upstream unavailable").count(),
                1
            );
            (
                failure.outcome.status,
                failure.outcome.stop_reason,
                failure.outcome.verification,
            )
        };
        if leftover && check_passes {
            assert_eq!(status, TurnStatus::Completed);
            assert_eq!(stop_reason, TurnStopReason::Completed);
        } else {
            assert_eq!(status, TurnStatus::Failed);
            assert_eq!(
                stop_reason,
                if leftover {
                    TurnStopReason::VerificationFailed
                } else {
                    TurnStopReason::InfrastructureFailure
                }
            );
            assert_eq!(
                ui.assistant.matches("Automatic recovery stopped").count(),
                1,
                "closeout must not be duplicated: {}",
                ui.assistant
            );
        }
        assert_eq!(
            verification,
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
        if leftover {
            assert_eq!(
                subject.task_recovery().interventions,
                0,
                "leftover budgets must not spend recovery interventions: {:?}",
                subject.task_recovery().last_reason
            );
            let reason = subject.task_recovery().last_reason.as_deref().unwrap_or("");
            match kind {
                TerminalProviderFailure::Protocol => {
                    assert!(
                        ui.statuses.iter().any(|status| {
                            status.contains("invalid tool turn")
                                && status.contains("tool-format guidance")
                        }),
                        "HTTP 400 tool_protocol retries must use the protocol path: {:?}",
                        ui.statuses
                    );
                    assert!(
                        reason.contains("invalid tool turns exhausted"),
                        "{reason:?}"
                    );
                }
                TerminalProviderFailure::Capacity => {
                    assert!(reason.contains("request limit exhausted"), "{reason:?}");
                    assert!(
                        ui.statuses.iter().any(|status| {
                            status.contains("request limit exhausted")
                                && status.contains("without treating it as a provider outage")
                        }),
                        "{:?}",
                        ui.statuses
                    );
                    assert!(
                        !ui.statuses.iter().any(|status| {
                            status.contains("infrastructure")
                                || status.contains("provider requests stopped")
                        }),
                        "{:?}",
                        ui.statuses
                    );
                    assert_ne!(stop_reason, TurnStopReason::InfrastructureFailure);
                }
                TerminalProviderFailure::Outage => {}
            }
        }
    }
}

#[tokio::test]
async fn review_and_fix_protocol_storm_recovers_without_draining_recovery() {
    // Live ~/chat stall: git status / cargo check, then 400 tool_protocol
    // JSON until 4/4 sends and recovery remaining hit 0. Protocol retries
    // must stay format steering so keep-working can still land an edit.
    let workspace = IsolatedWorkspace::new("review-fix-protocol-storm");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn f() {}\n").unwrap();
    let inspect = serde_json::json!({"choices":[{"delta":{"tool_calls":[{
        "index":0,"id":"st","type":"function","function":{
            "name":"bash",
            "arguments": serde_json::json!({"command":"printf '%s\\n' '?? src/'"}).to_string()
        }
    }]},"finish_reason":"tool_calls"}]});
    let edit = serde_json::json!({"choices":[{"delta":{"tool_calls":[{
        "index":0,"id":"ed","type":"function","function":{
            "name":"write",
            "arguments": serde_json::json!({"path":"src/lib.rs","content":"pub fn f() { 1 }\n"}).to_string()
        }
    }]},"finish_reason":"tool_calls"}]});
    let mut responses = vec![Response::sse(format!(
        "data: {inspect}\n\ndata: [DONE]\n\n"
    ))];
    responses.extend((0..4).map(|_| {
        Response::json(
            400,
            r#"{"error":{"message":"invalid tool JSON","code":"tool_protocol_error","retryable":true}}"#,
        )
    }));
    responses.push(Response::sse(format!("data: {edit}\n\ndata: [DONE]\n\n")));
    responses.push(Response::sse(sse_text("Fixed the empty function body.")));
    let Some(server) = FakeOpenAiServer::new(responses) else {
        return;
    };
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 1;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let provider = hi_ai::OpenAiProvider::new(server.url().to_owned(), "test".into());
    let mut subject = Agent::new(Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();
    let outcome = subject
        .run_turn("review for any major issues and fix", &mut ui)
        .await
        .expect("protocol storm after inspection must not become a provider outage");
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "live stall: {outcome:?}; statuses={:?}; last={:?}",
        ui.statuses,
        subject.task_recovery().last_reason
    );
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(
        subject
            .messages()
            .iter()
            .any(|message| message.text().contains("[hi:nudge:protocol]")),
        "HTTP tool_protocol retries must use the protocol nudge"
    );
    assert!(
        subject.task_recovery().interventions <= 1,
        "only keep-working may spend recovery, not protocol retries: interventions={} last={:?}",
        subject.task_recovery().interventions,
        subject.task_recovery().last_reason
    );
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn f() { 1 }\n"
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("provider outage") || status.contains("infrastructure")),
        "{:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn protocol_storm_after_keep_working_wraps_up_instead_of_no_progress() {
    // Live ~/chat 1788993673517: "build all of that", failed edit, keep-working,
    // then a second invalid-tool storm settled NoProgress with no recap.
    let workspace = IsolatedWorkspace::new("protocol-after-keep-working-wrap-up");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/db.rs"), "pub struct Db;\n").unwrap();
    let inspect = serde_json::json!({"choices":[{"delta":{"tool_calls":[{
        "index":0,"id":"rd","type":"function","function":{
            "name":"read",
            "arguments": serde_json::json!({"path":"src/db.rs"}).to_string()
        }
    }]},"finish_reason":"tool_calls"}]});
    let recap = "Persist DMs and deliver on login is unfinished. The last edit to src/db.rs failed because old_string did not match, and later tool calls were rejected as invalid JSON. No files were changed.";
    let mut responses = vec![Response::sse(format!(
        "data: {inspect}\n\ndata: [DONE]\n\n"
    ))];
    // Implementation routes spend one send on text-tool fallback in each
    // consecutive storm (5 invalid turns: 2 retries, fallback, 1 retry, settle).
    // Two storms: keep-working, then wrap-up.
    responses.extend((0..10).map(|_| {
        Response::json(
            400,
            r#"{"error":{"message":"invalid tool JSON","code":"tool_protocol_error","retryable":true}}"#,
        )
    }));
    responses.push(Response::sse(sse_text(recap)));
    let Some(server) = FakeOpenAiServer::new(responses) else {
        return;
    };
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = crate::MAX_KEEP_WORKING;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let provider = hi_ai::OpenAiProvider::new(server.url().to_owned(), "test".into());
    let mut subject = Agent::new(Arc::new(provider), cfg).unwrap();
    subject.restore_plan(vec![PlanStep {
        title: "Persist DMs and deliver on login".into(),
        status: PlanStatus::Active,
    }]);
    let mut ui = RecUi::default();
    let outcome = subject
        .run_turn("build all of that", &mut ui)
        .await
        .expect("protocol exhaustion after keep-working must not become a provider outage");
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "live stall: {outcome:?}; statuses={:?}; last={:?}; assistant={}",
        ui.statuses,
        subject.task_recovery().last_reason,
        ui.assistant
    );
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(
        ui.assistant.contains("Persist DMs") || ui.assistant.contains("unfinished"),
        "wrap-up recap missing: {}",
        ui.assistant
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for a final answer without tools")),
        "expected wrap-up after the second protocol storm: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("still working")),
        "keep-working should fire before wrap-up: {:?}",
        ui.statuses
    );
    assert!(
        subject
            .messages()
            .iter()
            .any(|message| message.text().contains("[hi:nudge:protocol]")),
        "HTTP tool_protocol retries must use the protocol nudge"
    );
    assert!(
        !subject.task_recovery().exhausted,
        "wrap-up must not absorb the remaining recovery budget: {:?}",
        subject.task_recovery().last_reason
    );
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/db.rs")).unwrap(),
        "pub struct Db;\n"
    );
    let last = server.bodies().last().cloned().unwrap_or_default();
    assert!(
        last.contains("\"tool_choice\":\"none\"")
            || last.contains("\"tools\":[]")
            || last.contains("\"tools\": []")
            || !last.contains("\"tools\""),
        "wrap-up request must be tool-free: {last}"
    );
}

#[tokio::test]
async fn protocol_storm_after_retained_edits_does_not_fail_as_no_progress() {
    // Live ~/chat 1788995710473: reply-queue + idle-timeout edits, cargo test
    // green, plan 2/7, then invalid tool JSON. Settlement verified the retained
    // workspace then branded the turn Failed/no_progress because recovery
    // stop()ed. Protocol after real work is leftover, not a stall.
    let workspace = IsolatedWorkspace::new("protocol-after-retained-edits");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn f() {}\n").unwrap();
    let edit = serde_json::json!({"choices":[{"delta":{"tool_calls":[{
        "index":0,"id":"ed","type":"function","function":{
            "name":"write",
            "arguments": serde_json::json!({"path":"src/lib.rs","content":"pub fn f() { 1 }\n"}).to_string()
        }
    }]},"finish_reason":"tool_calls"}]});
    let recap = "Idle timeout and the reply-queue fix are in. Per-channel rate limiting remains unfinished. cargo test was green before tool calls started being rejected as invalid JSON.";
    let mut responses = vec![Response::sse(format!("data: {edit}\n\ndata: [DONE]\n\n"))];
    responses.extend((0..10).map(|_| {
        Response::json(
            400,
            r#"{"error":{"message":"invalid tool JSON","code":"tool_protocol_error","retryable":true}}"#,
        )
    }));
    responses.push(Response::sse(sse_text(recap)));
    let Some(server) = FakeOpenAiServer::new(responses) else {
        return;
    };
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = crate::MAX_KEEP_WORKING;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let provider = hi_ai::OpenAiProvider::new(server.url().to_owned(), "test".into());
    let mut subject = Agent::new(Arc::new(provider), cfg).unwrap();
    subject.restore_plan(vec![
        PlanStep {
            title: "Fix reply drop-on-full queue".into(),
            status: PlanStatus::Done,
        },
        PlanStep {
            title: "Add idle timeout + connection cap".into(),
            status: PlanStatus::Done,
        },
        PlanStep {
            title: "Add per-channel rate limiting".into(),
            status: PlanStatus::Active,
        },
    ]);
    let mut ui = RecUi::default();
    let outcome = subject
        .run_turn("build all of that", &mut ui)
        .await
        .expect("protocol after retained edits is not a provider outage");
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "live stall: {outcome:?}; statuses={:?}; last={:?}; assistant={}",
        ui.statuses,
        subject.task_recovery().last_reason,
        ui.assistant
    );
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(
        subject.plan_incomplete(),
        "unfinished checklist steps remain leftover"
    );
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn f() { 1 }\n"
    );
    assert!(
        ui.assistant.contains("rate limiting") || ui.assistant.contains("unfinished"),
        "wrap-up recap missing: {}",
        ui.assistant
    );
    assert!(
        !ui.assistant.contains("No file changes were made"),
        "retained edits must not be reported as no file changes: {}",
        ui.assistant
    );
}
