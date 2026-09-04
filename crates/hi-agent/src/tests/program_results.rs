use super::common::*;
use super::*;

struct ProgramProvider {
    responses: Mutex<Vec<Completion>>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for ProgramProvider {
    async fn stream(
        &self,
        _: hi_ai::ChatRequest,
        _: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        pop_canned_completion(&self.responses, "ProgramProvider")
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        hi_ai::ProviderCapabilities::native_tools(false)
    }
}

#[tokio::test]
async fn program_result_keeps_selected_answer_without_echoing_all_source() {
    let workspace = IsolatedWorkspace::new("selected-program-result");
    std::fs::write(
        workspace.path("source.txt"),
        "a long source line that does not need repeating\n".repeat(500),
    )
    .unwrap();
    let mut cfg = workspace.config();
    cfg.program.mode = crate::ProgramMode::Auto;
    let mut agent = Agent::new(std::sync::Arc::new(ProgramProvider {
        responses: Mutex::new(vec![
            completion(vec![Content::ToolCall {
                id: "program".into(), name: "run_program".into(),
                arguments: serde_json::json!({"source": "let content = tool(\"read\", #{path: \"source.txt\"}); #{selected: content.output.contains(\"long source\")}"}).to_string(),
            }], 1, 1),
            completion(vec![Content::Text("The requested content is present.".into())], 1, 1),
        ]),
    }), cfg).unwrap();
    agent
        .run_turn(
            "Determine whether the source contains the phrase",
            &mut NullUi,
        )
        .await
        .unwrap();
    let output = agent
        .messages()
        .iter()
        .flat_map(|message| &message.content)
        .find_map(|block| match block {
            Content::ToolResult { call_id, output } if call_id == "program" => Some(output),
            _ => None,
        })
        .unwrap();
    let value: serde_json::Value =
        serde_json::from_str(output).expect("program result must remain valid JSON");
    assert_eq!(value["result"]["selected"], true);
    assert_eq!(value["calls"][0]["status"], "succeeded");
    assert!(
        output.len() < 1000,
        "selected result should not echo a full source file: {} bytes",
        output.len()
    );
}
