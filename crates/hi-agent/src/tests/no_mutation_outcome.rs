use super::common::{IsolatedWorkspace, RecordingUi, agent, completion};
use super::*;

fn repeated_read(id: &str) -> Content {
    Content::ToolCall {
        id: id.into(),
        name: "read".into(),
        arguments: "{\"path\":\"src/parser.rs\"}".into(),
    }
}

fn no_edit_agent(workspace: &IsolatedWorkspace, prefix: &str) -> Agent {
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/parser.rs"), "fn parse() {}\n").unwrap();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_repeat_nudges = 0;
    agent(
        vec![
            completion(
                vec![
                    Content::Text("I found the issue and will apply the fix next.".into()),
                    repeated_read(&format!("{prefix}-r1")),
                ],
                1,
                1,
            ),
            completion(vec![repeated_read(&format!("{prefix}-r2"))], 1, 1),
            completion(vec![repeated_read(&format!("{prefix}-r3"))], 1, 1),
            completion(vec![repeated_read(&format!("{prefix}-r4"))], 1, 1),
        ],
        cfg,
    )
}

#[tokio::test]
async fn explicit_mutation_request_without_changes_settles_as_no_progress() {
    let workspace = IsolatedWorkspace::new("outcome-explicit-no-changes");
    let mut agent = no_edit_agent(&workspace, "direct");
    let mut ui = RecordingUi::default();

    let outcome = agent.run_turn("fix the parser bug", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::NotApplicable);
    assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert!(ui.statuses.iter().any(|s| s.contains("no file changes")));
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| { message.text().contains("Automatic recovery stopped.") })
    );
}

#[tokio::test]
async fn review_and_fix_without_changes_settles_as_no_progress() {
    let workspace = IsolatedWorkspace::new("outcome-review-fix-no-changes");
    let mut agent = no_edit_agent(&workspace, "review");
    let mut ui = RecordingUi::default();

    let outcome = agent
        .run_turn("review the parser bug, dig in and fix it", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert!(agent.messages().iter().any(|message| {
        message
            .text()
            .contains("Implementation guard: inspect the workspace")
    }));
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| { message.text().contains("Automatic recovery stopped.") })
    );
}

#[tokio::test]
async fn bare_refusal_cannot_satisfy_an_explicit_mutation_request() {
    let workspace = IsolatedWorkspace::new("outcome-bare-mutation-refusal");
    let mut agent = agent(
        vec![
            completion(
                vec![Content::Text(
                    "I found the bug but have not edited it.".into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::Text("I won't modify the files.".into())],
                1,
                1,
            ),
            completion(vec![Content::Text("That is out of scope.".into())], 1, 1),
        ],
        workspace.config(),
    );

    let outcome = agent
        .run_turn("fix the parser bug", &mut RecordingUi::default())
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
}
