use super::common::{
    IsolatedWorkspace, NullUi, ProviderStep, bash_completion, completion, scripted_agent,
};
use super::*;

fn write_state(value: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: format!("state-{value}"),
            name: "write".into(),
            arguments: serde_json::json!({"path":"state.txt", "content":value}).to_string(),
        }],
        1,
        1,
    )
}

fn final_answer() -> Completion {
    completion(vec![Content::Text("The implementation is ready. I ran the checks and the focused suite passed. The source changes are saved in state.txt.".into())], 1, 1)
}

fn fixture(tag: &str) -> (IsolatedWorkspace, AgentConfig) {
    let workspace = IsolatedWorkspace::new(tag);
    std::fs::write(workspace.path("validate.py"),
        "import pathlib, sys\nfailed = '--all' in sys.argv and pathlib.Path('state.txt').read_text() != 'fixed'\nprint('test suite::broad_failure ... FAILED' if failed else 'test result: ok. 1 passed; 0 failed')\nsys.exit(1 if failed else 0)\n").unwrap();
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.allow_unverified = true;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    (workspace, cfg)
}

#[tokio::test]
async fn broad_failure_then_narrow_pass_and_accepted_prose_cannot_complete() {
    for mutate_after_failure in [false, true] {
        let (_workspace, cfg) = fixture("validation-completion-broad");
        let mut steps = vec![
            ProviderStep::Completion(write_state("broken")),
            ProviderStep::Completion(bash_completion("python3 validate.py --all")),
        ];
        if mutate_after_failure {
            steps.push(ProviderStep::Completion(write_state("changed")));
        }
        steps.extend([
            ProviderStep::Completion(bash_completion("python3 validate.py --focused")),
            ProviderStep::Completion(final_answer()),
        ]);
        let expected_requests = steps.len();
        let (mut agent, requests) = scripted_agent(steps, cfg);
        let outcome = agent
            .run_turn(
                "implement the requested change in state.txt and check it",
                &mut NullUi,
            )
            .await
            .unwrap();
        assert_eq!(requests.lock().unwrap().len(), expected_requests);
        assert_eq!(outcome.status, TurnStatus::Failed);
        assert_eq!(
            outcome.verification,
            if mutate_after_failure {
                VerificationStatus::Unverified
            } else {
                VerificationStatus::Failed
            }
        );
        assert_eq!(
            outcome.stop_reason,
            if mutate_after_failure {
                TurnStopReason::VerificationUnavailable
            } else {
                TurnStopReason::VerificationFailed
            }
        );
        assert!(agent.messages().iter().any(|message| {
            let text = message.text();
            text.contains("Unresolved check: python3 validate.py --all")
                && text.contains("suite::broad_failure")
        }));
    }
}

#[tokio::test]
async fn same_scope_passing_recheck_clears_the_current_failure() {
    let (_workspace, cfg) = fixture("validation-completion-repaired");
    let steps = vec![
        ProviderStep::Completion(write_state("broken")),
        ProviderStep::Completion(bash_completion("python3 validate.py --all")),
        ProviderStep::Completion(write_state("fixed")),
        ProviderStep::Completion(bash_completion("python3 validate.py --all")),
        ProviderStep::Completion(final_answer()),
    ];
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let outcome = agent
        .run_turn(
            "implement the requested change in state.txt and check it",
            &mut NullUi,
        )
        .await
        .unwrap();
    assert_eq!(requests.lock().unwrap().len(), 5);
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(
        agent
            .task_recovery()
            .unresolved_validation_summary("current")
            .is_none()
    );
}

#[tokio::test]
async fn quoted_whitespace_distinguishes_checks_and_cannot_clear_failure() {
    let (_workspace, cfg) = fixture("validation-quoted-scope");
    let failed = "python3 -c \"assert 'a  b' == 'a b'\"";
    let passed = "python3 -c \"assert 'a b' == 'a b'\"";
    let steps = vec![
        ProviderStep::Completion(write_state("broken")),
        ProviderStep::Completion(bash_completion(failed)),
        ProviderStep::Completion(bash_completion(passed)),
        ProviderStep::Completion(final_answer()),
    ];
    let (mut subject, requests) = scripted_agent(steps, cfg);
    let outcome = subject
        .run_turn(
            "implement the requested change in state.txt and check it",
            &mut NullUi,
        )
        .await
        .unwrap();
    assert_eq!(requests.lock().unwrap().len(), 4);
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::VerificationFailed);
    assert_eq!(
        subject.task_recovery().remaining,
        2,
        "distinct green check cannot replenish recovery"
    );
    assert!(
        subject
            .task_recovery()
            .unresolved_validation_summary("current")
            .unwrap()
            .contains(failed)
    );
}

struct GoalExportAtTurnEnd {
    path: std::path::PathBuf,
    overwrite: bool,
}
impl Ui for GoalExportAtTurnEnd {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {
        if self.overwrite {
            std::fs::write(&self.path, "later export").unwrap();
        }
    }
}

#[tokio::test]
async fn explicitly_requested_goal_export_remains_reported_and_verification_sensitive() {
    for overwrite in [false, true] {
        let workspace = IsolatedWorkspace::new("requested-goal-export");
        let mut cfg = workspace.config();
        cfg.gates.review = ReviewPolicy::Off;
        cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new(
            "goal export",
            "python3 -c \"from pathlib import Path; assert Path('.hi/goal-plan.md').read_text() == 'requested'\"",
        )]);
        let steps = vec![
            ProviderStep::Completion(completion(vec![Content::ToolCall {
                id: "requested-export".into(), name: "write".into(),
                arguments: serde_json::json!({"path":crate::goal::GOAL_EXPORT_PATH,"content":"requested"}).to_string(),
            }], 1, 1)),
            ProviderStep::Completion(completion(vec![Content::Text("Updated the goal export with the requested text and ran its exact-content check.".into())], 1, 1)),
        ];
        let (mut agent, _) = scripted_agent(steps, cfg);
        let mut ui = GoalExportAtTurnEnd {
            path: workspace.path(crate::goal::GOAL_EXPORT_PATH),
            overwrite,
        };
        let outcome = agent
            .run_turn(
                "write requested to .hi/goal-plan.md and verify its exact content",
                &mut ui,
            )
            .await
            .unwrap();
        assert!(
            outcome
                .changed_files
                .contains(&crate::goal::GOAL_EXPORT_PATH.to_owned())
        );
        if overwrite {
            assert_eq!(outcome.status, TurnStatus::Failed);
            assert_eq!(outcome.verification, VerificationStatus::Unverified);
        } else {
            assert_eq!(outcome.status, TurnStatus::Completed);
            assert_eq!(outcome.verification, VerificationStatus::Passed);
        }
    }
}

#[tokio::test]
async fn read_only_goal_export_validation_is_bound_to_the_checked_bytes() {
    for overwrite in [false, true] {
        let workspace = IsolatedWorkspace::new("checked-goal-export");
        std::fs::create_dir_all(workspace.path(".hi")).unwrap();
        std::fs::write(workspace.path(crate::goal::GOAL_EXPORT_PATH), "broken").unwrap();
        let cfg = workspace.config();
        let steps = vec![
            ProviderStep::Completion(bash_completion("python3 -c \"from pathlib import Path; assert Path('.hi/goal-plan.md').read_text() == 'requested'\"")),
            ProviderStep::Completion(completion(vec![Content::Text("The requested check failed: the existing goal export does not contain the required text.".into())], 1, 1)),
        ];
        let (mut agent, requests) = scripted_agent(steps, cfg);
        let mut ui = GoalExportAtTurnEnd {
            path: workspace.path(crate::goal::GOAL_EXPORT_PATH),
            overwrite,
        };
        let outcome = agent
            .run_turn(
                "check the exact text in .hi/goal-plan.md and report its result",
                &mut ui,
            )
            .await
            .unwrap();
        assert_eq!(requests.lock().unwrap().len(), 2);
        assert_eq!(outcome.status, TurnStatus::Failed);
        assert_eq!(
            outcome.verification,
            if overwrite {
                VerificationStatus::Unverified
            } else {
                VerificationStatus::Failed
            }
        );
        assert_eq!(
            outcome
                .changed_files
                .contains(&crate::goal::GOAL_EXPORT_PATH.to_owned()),
            overwrite,
            "registering an existing validation input must not invent a file edit"
        );
    }
}
