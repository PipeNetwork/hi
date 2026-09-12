use super::common::*;
use super::*;
use hi_ai::{Content, Message, Role};

fn bulky_history() -> Vec<Message> {
    let body = "x".repeat(8_000);
    vec![
        Message::system("sys"),
        Message::user("do the work"),
        Message {
            role: Role::Assistant,
            content: vec![Content::ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                arguments: r#"{"command":"echo"}"#.into(),
            }],
        },
        Message {
            role: Role::Tool,
            content: vec![Content::ToolResult {
                call_id: "c1".into(),
                output: body,
            }],
        },
        Message::user("continue the remaining plan"),
    ]
}

#[tokio::test]
async fn plan_boundary_window_pressure_compacts_and_reorients_next_turn() {
    let mut cfg = config();
    cfg.memory.online_context_compact = true;
    cfg.memory.auto_compact = true;
    cfg.memory.compaction = CompactionKind::ElideToolOutput { keep_recent: 1 };
    cfg.routing.context_window = Some(10_000);
    cfg.routing.max_tokens = 256;
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("reoriented and done".into())],
            10,
            8,
        )],
        cfg,
    );
    agent.messages = crate::transcript::Transcript::new(bulky_history());
    let _ = agent.goals.replace_plan(&[
        hi_tools::PlanStep {
            title: "done step".into(),
            status: hi_tools::PlanStatus::Done,
        },
        hi_tools::PlanStep {
            title: "next step".into(),
            status: hi_tools::PlanStatus::Pending,
        },
    ]);
    agent.online_compact.completed_boundary_request_counts = vec![1];
    agent.online_compact.record_boundary();
    agent.report.context_used = 9_500;

    let compacted = agent
        .maybe_plan_boundary_compact(&mut NullUi)
        .await
        .unwrap();
    assert!(compacted, "window pressure should compact");
    assert!(agent.online_compact.pending_reorient);

    let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let provider = ScriptedProvider {
        steps: std::sync::Mutex::new(vec![ProviderStep::Completion(completion(
            vec![Content::Text("still working".into())],
            10,
            8,
        ))]),
        requests: requests.clone(),
        max_tokens: None,
    };
    agent.provider = std::sync::Arc::new(provider);
    agent.run_turn("keep going", &mut NullUi).await.unwrap();
    let captured = requests.lock().unwrap();
    assert!(
        !captured.is_empty(),
        "next model-facing turn must still run"
    );
    let joined = captured
        .iter()
        .flatten()
        .map(|message| message.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        joined.contains("prior reads") || joined.contains("Re-orient"),
        "reorient missing from request: {joined}"
    );
}
