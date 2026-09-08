use super::common::*;
use super::*;
use hi_ai::Content;
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn first_waiting_poll_does_not_end_the_turn_on_a_status_line() {
    let provider = Arc::new(Canned(Mutex::new(Vec::new())));
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = crate::MAX_SILENT_CONTINUES;
    let mut agent = Agent::new(provider.clone(), cfg).unwrap();
    agent.runtime.background().set_poll_wait_base_secs(Some(0));
    let id = agent
        .runtime
        .background()
        .spawn(agent.runtime.process_runner(), "sleep 600")
        .unwrap();
    let bash_output = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: "bo".into(),
                name: "bash_output".into(),
                arguments: serde_json::json!({ "id": id }).to_string(),
            }],
            1,
            1,
        )
    };
    provider.0.lock().unwrap().extend(vec![
        completion(
            vec![Content::ToolCall {
                id: "plan".into(),
                name: "update_plan".into(),
                arguments: serde_json::json!({
                    "steps": [
                        { "title": "Watch the download", "status": "active" },
                        { "title": "Convert the file", "status": "pending" },
                    ]
                })
                .to_string(),
            }],
            1,
            1,
        ),
        bash_output(&id),
        completion(
            vec![Content::Text(
                "I will convert the file when the download finishes.".into(),
            )],
            1,
            1,
        ),
        bash_output(&id),
        bash_output(&id),
        completion(
            vec![Content::Text(
                "Work remains in progress: the download is still running; conversion has not started.".into(),
            )],
            1,
            1,
        ),
    ]);
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("watch the download and report status", &mut ui)
        .await
        .unwrap();
    let _ = agent.runtime.background().kill(&id);
    let bash_output_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "bash_output")
        .count();
    assert!(
        bash_output_results >= 2,
        "one poll plus a status line must keep tools available: {:?}",
        ui.tool_results
    );
    assert_ne!(
        outcome.stop_reason,
        crate::TurnStopReason::InfrastructureFailure,
        "statuses={:?}; leftover={:?}",
        ui.statuses,
        provider.0.lock().unwrap().len()
    );
}
