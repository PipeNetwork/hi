use super::common::{
    IsolatedWorkspace, ProviderStep, RecUi, completion, scripted_agent, write_content_completion,
};
use super::*;
use hi_ai::Content;

fn read_completion(path: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: "r".into(),
            name: "read".into(),
            arguments: format!(r#"{{"path":{path:?}}}"#),
        }],
        1,
        1,
    )
}

#[tokio::test]
async fn identical_reads_after_write_end_as_stationarity_not_no_progress() {
    let workspace = IsolatedWorkspace::new("stationarity-after-write");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn f() {}\n").unwrap();
    let path = workspace.path("src/lib.rs").to_string_lossy().into_owned();
    let mut steps = vec![ProviderStep::Completion(write_content_completion(
        &path,
        "pub fn f() { 1 }\n",
    ))];
    for _ in 0..crate::steering::MAX_CONSECUTIVE_IDENTICAL_PROBLEMATIC_TOOL_CALLS {
        steps.push(ProviderStep::Completion(read_completion(&path)));
    }
    steps.push(ProviderStep::Completion(completion(
        vec![Content::Text("Stopped repeating the read.".into())],
        1,
        1,
    )));
    let mut cfg = workspace.config();
    cfg.loop_limits.max_repeat_nudges = 16;
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let (mut agent, _) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("build all of that", &mut ui).await.unwrap();
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "stationarity after edits must not be no_progress: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn f() { 1 }\n"
    );
}

fn last_request_text(request: &[hi_ai::Message]) -> String {
    request.last().map(hi_ai::Message::text).unwrap_or_default()
}

#[tokio::test]
async fn false_completion_claim_without_tool_evidence_injects_laziness() {
    let workspace = IsolatedWorkspace::new("false-completion-laziness");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn f() {}\n").unwrap();
    let path = workspace.path("src/lib.rs").to_string_lossy().into_owned();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "SUCCESS: cargo test --quiet is all green and production-ready.".into(),
                )],
                1,
                1,
            )),
            ProviderStep::Completion(write_content_completion(&path, "pub fn f() { 1 }\n")),
            ProviderStep::Completion(completion(
                vec![Content::Text("Wrote the function body.".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("build all of that", &mut ui).await.unwrap();
    let sent = requests.lock().unwrap();
    assert!(
        sent.len() >= 2,
        "laziness continue must send another provider request: {}",
        sent.len()
    );
    let follow_up = last_request_text(&sent[1]);
    assert_eq!(
        sent[1].last().map(|message| message.role),
        Some(hi_ai::Role::User),
        "laziness nudge must be the last message on the next request"
    );
    assert!(
        follow_up.contains("[hi:nudge:laziness]"),
        "next provider request must end on the laziness user nudge, got: {follow_up}"
    );
    assert!(
        !follow_up.contains("SUCCESS: cargo test"),
        "false-completion claim must not be folded into the laziness user message: {follow_up}"
    );
    assert_ne!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert_ne!(outcome.stop_reason, TurnStopReason::InfrastructureFailure);
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn f() { 1 }\n"
    );
}

#[tokio::test]
async fn write_then_false_completion_without_tests_injects_laziness() {
    let workspace = IsolatedWorkspace::new("write-then-false-completion");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn f() {}\n").unwrap();
    let path = workspace.path("src/lib.rs").to_string_lossy().into_owned();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(&path, "pub fn f() { 1 }\n")),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "SUCCESS: cargo test --quiet is all green and production-ready.".into(),
                )],
                1,
                1,
            )),
            ProviderStep::Completion(super::common::bash_completion(
                "python3 -c 'assert 2 + 2 == 4'",
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "Edited src/lib.rs and ran a local check.".into(),
                )],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("build all of that", &mut ui).await.unwrap();
    let sent = requests.lock().unwrap();
    assert!(
        sent.len() >= 3,
        "write then false completion must continue after laziness: {}",
        sent.len()
    );
    let follow_up = last_request_text(&sent[2]);
    assert_eq!(
        sent[2].last().map(|message| message.role),
        Some(hi_ai::Role::User),
        "laziness nudge must be the last message after write-then-false-completion"
    );
    assert!(
        follow_up.contains("[hi:nudge:laziness]"),
        "next provider request after the false recap must be the laziness user nudge, got: {follow_up}"
    );
    assert_ne!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn f() { 1 }\n"
    );
}

#[tokio::test]
async fn todo_gate_continues_then_falls_through_without_keep_working() {
    let mut cfg = super::common::config();
    cfg.loop_limits.max_silent_continues = 1;
    cfg.loop_limits.max_keep_working = 2;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    let plan_call = |id: &str, s1: &str, s2: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"Review a","status":"{s1}"}},{{"title":"Review b","status":"{s2}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(plan_call("p1", "active", "pending")),
            ProviderStep::Completion(completion(
                vec![Content::Text("Step 1 recap.".into())],
                1,
                1,
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text("Still recapping leftover work.".into())],
                1,
                1,
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text("Would have been keep-working.".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("implement the remaining plan step", &mut ui)
        .await
        .unwrap();
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| message.text().contains("[hi:nudge:todogate]")),
        "TodoGate reminder missing: {:?}",
        agent
            .messages()
            .iter()
            .map(|m| m.text())
            .collect::<Vec<_>>()
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("still working")),
        "keep-working must not stack on TodoGate: {:?}",
        ui.statuses
    );
    assert_ne!(outcome.stop_reason, TurnStopReason::InfrastructureFailure);
    assert!(agent.plan_incomplete());
}

#[tokio::test]
async fn exhausted_implementation_todo_gate_retains_evidence_without_renewing_work() {
    for mutates in [false, true] {
        let workspace = IsolatedWorkspace::new(if mutates {
            "todo-gate-retains-mutation"
        } else {
            "todo-gate-stops-inspection"
        });
        let path = workspace.path("implementation.txt");
        std::fs::write(&path, "existing implementation\n").unwrap();
        let path = path.to_string_lossy().into_owned();
        let mut cfg = workspace.config();
        cfg.loop_limits.max_silent_continues = 1;
        cfg.loop_limits.max_keep_working = 2;
        cfg.gates.allow_unverified = true;
        cfg.gates.verification = VerificationMode::Disabled;
        cfg.gates.review = ReviewPolicy::Off;
        let tool = if mutates {
            write_content_completion(&path, "retained implementation\n")
        } else {
            read_completion(&path)
        };
        let (mut agent, requests) = scripted_agent(
            vec![
                ProviderStep::Completion(tool),
                ProviderStep::Completion(completion(
                    vec![Content::Text(
                        "The targeted implementation was inspected.".into(),
                    )],
                    1,
                    1,
                )),
                ProviderStep::Completion(completion(
                    vec![Content::Text(
                        "The next implementation step is still pending.".into(),
                    )],
                    1,
                    1,
                )),
            ],
            cfg,
        );
        agent.restore_plan(vec![crate::PlanStep {
            title: "Implement the remaining lifecycle marker".into(),
            status: crate::PlanStatus::Pending,
        }]);
        let mut ui = RecUi::default();
        let outcome = agent
            .run_turn("implement the remaining plan step", &mut ui)
            .await
            .unwrap();

        assert_eq!(requests.lock().unwrap().len(), 3, "{outcome:?}");
        assert!(agent.plan_incomplete());
        if mutates {
            assert_eq!(outcome.status, TurnStatus::Completed, "{outcome:?}");
            assert_ne!(outcome.stop_reason, TurnStopReason::NoProgress);
            assert_eq!(
                std::fs::read_to_string(&path).unwrap(),
                "retained implementation\n"
            );
        } else {
            assert_eq!(outcome.status, TurnStatus::Failed, "{outcome:?}");
            assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
            assert!(outcome.changed_files.is_empty());
            assert!(
                agent
                    .messages()
                    .iter()
                    .flat_map(|message| &message.content)
                    .any(|content| matches!(content, Content::ToolResult { output, .. } if output.contains("existing implementation")))
            );
        }
    }
}
