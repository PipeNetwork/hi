use std::sync::{Arc, Mutex};

use hi_ai::Content;
use hi_workspace::WorkspaceState;

use super::common::{Canned, RecUi, completion, config};

fn call(id: &str, name: &str, arguments: serde_json::Value) -> hi_ai::Completion {
    completion(
        vec![Content::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.to_string(),
        }],
        1,
        1,
    )
}

#[tokio::test]
async fn terminal_bash_output_settles_live_writer_before_publication() {
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "spawn",
            "bash",
            serde_json::json!({
                "command": "sleep 600",
                "run_in_background": true,
            }),
        ),
        completion(
            vec![Content::Text("Background process started.".into())],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider.clone(), config()).unwrap();
    let mut ui = RecUi::default();

    subject.run_turn("start a service", &mut ui).await.unwrap();

    let ids = subject.runtime.background().ids();
    assert_eq!(ids.len(), 1);
    assert_eq!(subject.workspace_controller_status().active_jobs.len(), 1);
    subject
        .runtime
        .background()
        .kill_and_reap(&ids[0])
        .await
        .unwrap();
    assert_eq!(
        subject.workspace_controller_status().active_jobs.len(),
        1,
        "reaping alone must leave the writer durability-pending"
    );

    provider.0.lock().unwrap().extend([
        call("poll", "bash_output", serde_json::json!({ "id": ids[0] })),
        completion(vec![Content::Text("The process was stopped.".into())], 1, 1),
    ]);
    subject
        .run_turn("check the service", &mut ui)
        .await
        .unwrap();

    let status = subject.workspace_controller_status();
    assert_eq!(status.state, WorkspaceState::Ready);
    assert!(
        status.active_jobs.is_empty(),
        "the terminal poll must not publish before its workspace receipt seals the job"
    );
    assert!(
        ui.tool_results
            .iter()
            .any(|(name, result)| name == "bash_output"
                && result.to_ascii_lowercase().contains("stopped")),
        "tool results: {:?}",
        ui.tool_results
    );
}

#[tokio::test]
async fn bash_kill_reaps_then_reconciles_the_existing_writer_job() {
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "spawn",
            "bash",
            serde_json::json!({
                "command": "sleep 600",
                "run_in_background": true,
            }),
        ),
        completion(
            vec![Content::Text("Background process started.".into())],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider.clone(), config()).unwrap();
    let mut ui = RecUi::default();
    subject.run_turn("start a service", &mut ui).await.unwrap();

    let ids = subject.runtime.background().ids();
    assert_eq!(ids.len(), 1);
    assert_eq!(subject.workspace_controller_status().active_jobs.len(), 1);
    provider.0.lock().unwrap().extend([
        call("kill", "bash_kill", serde_json::json!({ "id": ids[0] })),
        completion(vec![Content::Text("The process was stopped.".into())], 1, 1),
    ]);

    subject.run_turn("stop the service", &mut ui).await.unwrap();

    let status = subject.workspace_controller_status();
    assert_eq!(status.state, WorkspaceState::Ready);
    assert!(
        status.active_jobs.is_empty(),
        "kill must reap first, then reconcile and seal the existing writer job"
    );
    assert!(ui.tool_results.iter().any(|(name, result)| {
        name == "bash_kill" && result.to_ascii_lowercase().contains("stopped")
    }));
}

#[tokio::test]
async fn orderly_exit_reaps_processes_settles_tasks_and_passes_the_barrier() {
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "spawn",
            "bash",
            serde_json::json!({
                "command": "printf shutdown > shutdown-writer.txt; sleep 600",
                "run_in_background": true,
            }),
        ),
        completion(
            vec![Content::Text("Background process started.".into())],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider, config()).unwrap();
    let mut ui = RecUi::default();
    subject.run_turn("start a writer", &mut ui).await.unwrap();

    let tasks = subject.background_task_registry();
    let task_id = tasks
        .spawn(
            "pending reader",
            "explore",
            Box::new(|| Box::pin(std::future::pending::<hi_tools::BackgroundTaskOutcome>())),
        )
        .await
        .unwrap();
    assert_eq!(subject.workspace_controller_status().active_jobs.len(), 2);

    let receipt = subject.settle_workspace_for_exit().await.unwrap();

    assert_eq!(receipt.status, hi_workspace::BarrierStatus::Passed);
    assert!(subject.active_background_process_ids().is_empty());
    assert!(subject.active_background_task_ids().await.is_empty());
    assert!(subject.workspace_controller_status().active_jobs.is_empty());
    assert_eq!(
        tasks
            .poll(&task_id, std::time::Duration::ZERO)
            .await
            .unwrap()
            .state,
        hi_tools::BackgroundTaskState::Cancelled
    );
}
