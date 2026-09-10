use super::common::*;
use hi_tools::{BackgroundTaskOutcome, BackgroundTaskState};
use std::time::Duration;

#[tokio::test]
async fn wait_tasks_aborts_when_a_follow_up_arrives() {
    let agent = agent(Vec::new(), config());
    let id = agent
        .bg_tasks
        .spawn(
            "hang",
            "explore",
            Box::new(|| {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    BackgroundTaskOutcome {
                        id: String::new(),
                        description: "hang".into(),
                        subagent_type: "explore".into(),
                        state: BackgroundTaskState::Completed,
                        output: "done".into(),
                        applied: false,
                        changed_files: vec![],
                    }
                })
            }),
        )
        .await
        .unwrap();
    let started = std::time::Instant::now();
    let args = format!(r#"{{"task_ids":["{id}"],"mode":"wait_all","timeout_ms":8000}}"#);
    let wait = agent.handle_wait_tasks(&args);
    let inbox = agent.interjection_inbox();
    let push = async move {
        tokio::time::sleep(Duration::from_millis(40)).await;
        inbox.push("please stop waiting");
    };
    let (outcome, _) = tokio::join!(wait, push);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "follow-up must abort wait_tasks instead of parking on the timeout"
    );
    assert!(
        outcome.content.contains("follow-up") || outcome.content.contains("interrupted"),
        "interrupted wait must tell the model why: {}",
        outcome.content
    );
    let _ = agent.bg_tasks.kill(&id).await;
}
