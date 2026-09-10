use super::common::{
    IsolatedWorkspace, ProviderStep, RecordingUi, agent, bash_completion, completion,
    scripted_agent,
};
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

#[tokio::test]
async fn review_and_fix_bash_cat_of_read_files_still_lets_an_edit_land() {
    let workspace = IsolatedWorkspace::new("review-fix-bash-cat");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/a.rs"), "fn a() {}\n").unwrap();
    std::fs::write(workspace.path("src/b.rs"), "fn b() {}\n").unwrap();
    let read = |path: &str, id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": path }).to_string(),
            }],
            1,
            1,
        )
    };
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.allow_unverified = true;
    let mut subject = agent(
        vec![
            read("src/a.rs", "r1"),
            read("src/b.rs", "r2"),
            bash_completion("cd . && cat src/a.rs"),
            bash_completion("cd . && cat src/b.rs"),
            bash_completion("cd . && sed -n '1,20p' src/a.rs"),
            completion(
                vec![Content::ToolCall {
                    id: "w".into(),
                    name: "write".into(),
                    arguments: serde_json::json!({
                        "path": "src/a.rs",
                        "content": "fn a() { /* fixed */ }\n"
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text("Fixed src/a.rs.".into())], 1, 1),
        ],
        cfg,
    );
    let outcome = subject
        .run_turn(
            "review for any major issues and fix",
            &mut RecordingUi::default(),
        )
        .await
        .unwrap();
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
    assert!(
        std::fs::read_to_string(workspace.path("src/a.rs"))
            .unwrap()
            .contains("fixed"),
        "bash cat of already-read files must not exhaust recovery before an edit"
    );
}

#[tokio::test]
async fn review_and_fix_rereads_do_not_exhaust_recovery_before_an_edit() {
    let workspace = IsolatedWorkspace::new("review-fix-reread-recovery");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/a.rs"), "fn a() {}\n").unwrap();
    std::fs::write(workspace.path("src/b.rs"), "fn b() {}\n").unwrap();
    let read = |path: &str, id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": path }).to_string(),
            }],
            1,
            1,
        )
    };
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.allow_unverified = true;
    let mut subject = agent(
        vec![
            read("src/a.rs", "r1"),
            read("src/b.rs", "r2"),
            read("src/a.rs", "r3"),
            read("src/b.rs", "r4"),
            read("src/a.rs", "r5"),
            completion(
                vec![Content::ToolCall {
                    id: "w".into(),
                    name: "write".into(),
                    arguments: serde_json::json!({
                        "path": "src/a.rs",
                        "content": "fn a() { /* fixed */ }\n"
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text("Fixed src/a.rs.".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecordingUi::default();
    let outcome = subject
        .run_turn("review for any major issues and fix", &mut ui)
        .await
        .unwrap();
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}; statuses={:?}; remaining={}",
        ui.statuses,
        subject.task_recovery().remaining
    );
    assert!(
        std::fs::read_to_string(workspace.path("src/a.rs"))
            .unwrap()
            .contains("fixed"),
        "the edit must land instead of recovery exhausting on rereads"
    );
}

#[tokio::test]
async fn how_can_we_improve_review_is_not_no_progress() {
    let workspace = IsolatedWorkspace::new("improve-review-not-stall");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/a.rs"), "fn a() {}\n").unwrap();
    std::fs::write(workspace.path("src/b.rs"), "fn b() {}\n").unwrap();
    let read = |path: &str, id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": path }).to_string(),
            }],
            1,
            1,
        )
    };
    let review = "I have all the source in context now. Here's my review of the codebase \
and the improvements I'd recommend.\n\n\
## Major issues found\n\n\
1. PRIVMSG leaks whether a username exists (user enumeration).\n\
2. NICK rename leaks rate-limiter entries in the shared map.\n\
3. KICK broadcasts KICKED to every subscriber, including bystanders.\n\n\
Those are the highest-value, lowest-risk fixes.\n\n\
Let me implement fixes for #1 (user enumeration), #2 (rate-limiter leak), and #3 (KICKED broadcast).";
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.allow_unverified = true;
    let mut subject = agent(
        vec![
            read("src/a.rs", "r1"),
            read("src/b.rs", "r2"),
            read("src/a.rs", "r3"),
            read("src/b.rs", "r4"),
            completion(vec![Content::Text(review.into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecordingUi::default();
    assert_eq!(
        crate::GoalKind::derive("how can we imrpove this", false),
        crate::GoalKind::Analysis
    );
    let outcome = subject
        .run_turn("how can we imrpove this", &mut ui)
        .await
        .unwrap();
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "a delivered review must not fail as no_progress: {outcome:?}; statuses={:?}",
        ui.statuses
    );
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "{outcome:?}; statuses={:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn analysis_review_is_not_held_by_a_stale_checklist() {
    let workspace = IsolatedWorkspace::new("improve-stale-plan");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/a.rs"), "fn a() {}\n").unwrap();
    let review = "Here is the review.\n\n1. Close the NICK leak.\n2. Stop broadcasting KICKED.\n\n\
Let me implement those two fixes.";
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.allow_unverified = true;
    cfg.loop_limits.max_silent_continues = crate::MAX_SILENT_CONTINUES;
    let mut subject = agent(
        vec![completion(vec![Content::Text(review.into())], 1, 1)],
        cfg,
    );
    subject.goals.last_plan = vec![hi_tools::PlanStep {
        title: "fix the parser".into(),
        status: hi_tools::PlanStatus::Pending,
    }];
    let mut ui = RecordingUi::default();
    let outcome = subject
        .run_turn("how can we improve this", &mut ui)
        .await
        .unwrap();
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "analysis must accept the review instead of PLAN_CONTINUE_NUDGE: {outcome:?}; statuses={:?}",
        ui.statuses
    );
    assert_ne!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert!(
        !subject
            .messages()
            .iter()
            .any(|message| message.text().contains(crate::PLAN_CONTINUE_NUDGE)),
        "a leftover code-change checklist must not keep an analysis turn alive"
    );
}

#[tokio::test]
async fn review_and_fix_forced_final_after_repeated_checks_can_wrap_up() {
    // Live ~/chat stall: `cd && cargo clippy | grep` variants, then a leftover
    // "Fix issues" plan made the forced wrap-up unusable. Grok-build forces a
    // final to *stop*; the written answer (or green checks) is the deliverable.
    let workspace = IsolatedWorkspace::new("review-fix-wrap");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn f() {}\n").unwrap();
    let mut responses = vec![completion(
        vec![Content::ToolCall {
            id: "plan".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [
                    {"title": "Orient and read key source files", "status": "done"},
                    {"title": "Fix issues", "status": "pending"},
                    {"title": "Verify with cargo test", "status": "pending"}
                ]
            })
            .to_string(),
        }],
        1,
        1,
    )];
    for line in 1..=3 {
        responses.push(bash_completion(&format!(
            "python3 -c 'assert 2 + 2 == 4' 2>&1 | grep 'src/lib.rs:{line}' | head -40"
        )));
    }
    responses.push(completion(
        vec![Content::Text(
            "No major issues. The existing checks already pass.".into(),
        )],
        1,
        1,
    ));
    let mut cfg = workspace.config();
    cfg.loop_limits.max_repeat_nudges = 2;
    cfg.loop_limits.max_keep_working = 0;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.allow_unverified = true;
    let mut subject = agent(responses, cfg);
    let mut ui = RecordingUi::default();
    let outcome = subject
        .run_turn("review for any major issues and fix ", &mut ui)
        .await
        .unwrap();
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "forced wrap-up after repeated checks must not stall: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_eq!(outcome.status, TurnStatus::Completed);
}

#[tokio::test]
async fn code_change_bail_out_keeps_working_when_checklist_remains() {
    let workspace = IsolatedWorkspace::new("bail-out-continue");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/parser.rs"), "fn parse() {}\n").unwrap();
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.allow_unverified = true;
    cfg.loop_limits.max_silent_continues = 1;
    let mut subject = agent(
        vec![
            completion(vec![Content::Text("I can't proceed.".into())], 1, 1),
            completion(
                vec![Content::Text("The parser still needs an edit.".into())],
                1,
                1,
            ),
        ],
        cfg,
    );
    subject.goals.last_plan = vec![hi_tools::PlanStep {
        title: "fix the parser".into(),
        status: hi_tools::PlanStatus::Pending,
    }];
    let mut ui = RecordingUi::default();
    let _ = subject.run_turn("fix the parser bug", &mut ui).await;
    assert!(
        subject.messages().iter().any(|message| message
            .text()
            .contains(crate::steering::BAIL_CONTINUE_NUDGE)),
        "grok-build last-paragraph bail-out must continue code-change work: {:?}",
        subject
            .messages()
            .iter()
            .map(|message| message.text())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn invalid_tools_without_edits_are_no_progress_not_infrastructure() {
    let workspace = IsolatedWorkspace::new("protocol-no-edit");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/parser.rs"), "fn parse() {}\n").unwrap();
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.allow_unverified = true;
    cfg.loop_limits.max_keep_working = 0;
    let (mut subject, _) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(
                vec![Content::ToolCall {
                    id: "r1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"src/parser.rs"}"#.into(),
                }],
                1,
                1,
            )),
            ProviderStep::Error(hi_ai::ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(hi_ai::ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(hi_ai::ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(hi_ai::ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(hi_ai::ProviderErrorKind::ToolProtocol),
        ],
        cfg,
    );
    let mut ui = RecordingUi::default();
    let outcome = subject
        .run_turn("fix the parser bug", &mut ui)
        .await
        .expect("invalid tools must settle as a normal failed turn");
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
    let closeouts = subject
        .messages()
        .iter()
        .filter(|message| message.text().contains("Automatic recovery stopped"))
        .count();
    assert_eq!(closeouts, 1, "closeout must not be duplicated");
    assert!(
        subject
            .messages()
            .iter()
            .any(|message| message.text().contains("No file changes were made")),
        "{:?}",
        subject
            .messages()
            .iter()
            .map(|message| message.text())
            .collect::<Vec<_>>()
    );
}

#[test]
fn bash_cat_of_an_already_read_file_is_not_new_evidence() {
    let mut evidence = crate::steering::EvidenceTracker::default();
    evidence.record_success(
        "read",
        r#"{"path":"src/state.rs"}"#,
        "1\tuse std::sync::Arc;\n",
    );
    let cat =
        serde_json::json!({"command": "cd /Users/david/chat && cat src/state.rs"}).to_string();
    assert!(
        !evidence.round_adds_evidence(&[("c".into(), "bash".into(), cat)]),
        "dumping an already-read file via bash must not reset the inspection streak"
    );
    let unread = serde_json::json!({"command": "cat src/new.rs"}).to_string();
    assert!(evidence.round_adds_evidence(&[("c".into(), "bash".into(), unread)]));
}

#[test]
fn run_tests_to_verify_a_noun_change_is_not_an_edit_request() {
    let verify = TaskContract::derive(
        "Run cargo test to verify the rate limiter change didn't break anything.",
        VerificationMode::Auto,
    );
    assert!(verify.wants_tests);
    assert!(
        !verify.explicit_mutation,
        "noun 'change' must not demand file edits: {verify:?}"
    );

    let edit = TaskContract::derive("change the rate limiter", VerificationMode::Auto);
    assert!(edit.explicit_mutation);
}

#[tokio::test]
async fn piped_cargo_test_to_verify_a_noun_change_completes() {
    let workspace = IsolatedWorkspace::new("outcome-verify-noun-change");
    std::fs::create_dir(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"verify_noun_change\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    std::fs::write(
        workspace.path("src/lib.rs"),
        "#[test] fn answer() { assert_eq!(2 + 2, 4); }\n",
    )
    .unwrap();
    let mut cfg = workspace.config();
    cfg.gates.allow_unverified = true;
    let mut subject = agent(
        vec![
            bash_completion("cargo test --quiet 2>&1 | tail -30"),
            completion(
                vec![Content::Text(
                    "cargo test passed; no file changes are needed.".into(),
                )],
                1,
                1,
            ),
        ],
        cfg,
    );
    let mut ui = RecordingUi::default();
    let outcome = subject
        .run_turn(
            "Run cargo test to verify the rate limiter change didn't break anything.",
            &mut ui,
        )
        .await
        .unwrap();
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "{outcome:?}; {:?}; {:?}",
        ui.statuses,
        subject.last_turn_telemetry()
    );
    assert_ne!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert!(
        !ui.statuses.iter().any(|s| s.contains("no file changes")),
        "verify-only turn must not be challenged as a missing edit: {:?}",
        ui.statuses
    );
}
