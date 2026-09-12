use super::*;

struct NativeProgramProvider {
    responses: Mutex<Vec<Completion>>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for NativeProgramProvider {
    async fn stream(
        &self,
        _request: hi_ai::ChatRequest,
        _sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        pop_canned_completion(&self.responses, "NativeProgramProvider")
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        hi_ai::ProviderCapabilities::native_tools(false)
    }
}

#[derive(Default)]
struct SlowProgramConfirmationUi {
    confirmations: usize,
    tool_results: Vec<(String, String)>,
    confirmation_started: std::sync::Arc<tokio::sync::Notify>,
}

impl Ui for SlowProgramConfirmationUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}

    fn confirm(&mut self, _: crate::ConfirmationRequest) -> crate::ConfirmationFuture<'_> {
        self.confirmations += 1;
        let confirmation_started = self.confirmation_started.clone();
        Box::pin(async move {
            confirmation_started.notify_one();
            tokio::time::sleep(std::time::Duration::from_secs(61)).await;
            crate::ConfirmationResult::Rejected
        })
    }

    fn tool_call(&mut self, _: &str, _: &str) {}

    fn tool_result(&mut self, name: &str, result: &str) {
        self.tool_results
            .push((name.to_string(), result.to_string()));
    }

    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

#[tokio::test(start_paused = true)]
async fn productive_program_host_waits_past_the_legacy_total_deadline() {
    let mut cfg = config();
    cfg.program.mode = crate::ProgramMode::Auto;
    cfg.gates.confirm_edits = true;
    let provider = NativeProgramProvider {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "program".into(),
                    name: "run_program".into(),
                    arguments: serde_json::json!({
                        "source": r#"tool("web_fetch", #{url: "https://example.invalid"}); "finished""#
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "The workflow program completed after the host decision.".into(),
                )],
                1,
                1,
            ),
        ]),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    agent.set_permission_mode(crate::PermissionMode::Ask);
    let confirmation_started = std::sync::Arc::new(tokio::sync::Notify::new());
    let mut ui = SlowProgramConfirmationUi {
        confirmation_started: confirmation_started.clone(),
        ..SlowProgramConfirmationUi::default()
    };

    let outcome = {
        let turn = agent.run_turn("run the workflow", &mut ui);
        tokio::pin!(turn);
        tokio::select! {
            () = confirmation_started.notified() => {}
            result = &mut turn => panic!("turn settled before the delayed host decision: {result:?}"),
        }
        tokio::time::advance(std::time::Duration::from_secs(61)).await;
        // Durability now runs on a real writer thread; only approval uses virtual time.
        tokio::time::resume();
        turn.await.unwrap()
    };

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(ui.confirmations, 1);
    let program_result = ui
        .tool_results
        .iter()
        .find(|(name, _)| name == "run_program")
        .expect("program result was emitted");
    assert!(
        program_result.1.contains(r#""status":"succeeded""#),
        "a slow host decision must not fail the whole program: {program_result:?}"
    );
    assert!(!program_result.1.contains("total time budget"));
}
