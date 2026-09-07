//! Late callbacks cannot publish an outcome for an obsolete workspace input.

use super::common::*;
use super::*;
use std::sync::Arc;

struct LateUi {
    mutation: Option<std::path::PathBuf>,
    events: Vec<hi_events::EventKind>,
    statuses: Vec<String>,
}
impl Ui for LateUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, status: &str) {
        self.statuses.push(status.to_owned());
    }
    fn turn_end(&mut self, _: &str) {}
    fn suggested_prompt(&mut self, _: &str) {
        if let Some(path) = self.mutation.take() {
            std::fs::write(path, "pub fn unverified_callback() {}\n").unwrap();
        }
    }
    fn semantic_event(&mut self, event: hi_events::RunEvent) {
        self.events.push(event.kind);
    }
}

#[derive(Default)]
struct Records {
    fail_receipt: bool,
    outcomes: Vec<TurnOutcome>,
    recovery: Vec<TaskRecoveryState>,
    goals: Vec<Goal>,
}
struct TerminalSession(Arc<Mutex<Records>>);
impl SessionSink for TerminalSession {
    fn record(&mut self, _: &[Message], _: Usage) -> anyhow::Result<()> {
        Ok(())
    }
    fn record_compaction(&mut self, _: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }
    fn record_goal(&mut self, goal: &Goal) -> anyhow::Result<()> {
        let mut records = self.0.lock().unwrap();
        if goal
            .sub_goals
            .first()
            .is_some_and(|step| step.status == GoalStatus::Done)
        {
            anyhow::ensure!(
                !records.outcomes.is_empty(),
                "goal completion escaped before its settlement receipt"
            );
        }
        records.goals.push(goal.clone());
        Ok(())
    }
    fn record_task_recovery(&mut self, recovery: &TaskRecoveryState) -> anyhow::Result<()> {
        self.0.lock().unwrap().recovery.push(recovery.clone());
        Ok(())
    }
    fn record_turn_outcome(
        &mut self,
        outcome: &TurnOutcome,
        _: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut records = self.0.lock().unwrap();
        anyhow::ensure!(!records.fail_receipt, "terminal receipt unavailable");
        records.outcomes.push(outcome.clone());
        Ok(())
    }
}

fn verified_config(workspace: &IsolatedWorkspace) -> AgentConfig {
    let mut cfg = workspace.config();
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("check", "true")]);
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    cfg
}

fn code_responses() -> Vec<Completion> {
    vec![
        completion(
            vec![Content::ToolCall {
                id: "patch".into(),
                name: "apply_patch".into(),
                arguments: serde_json::json!({
                    "patch": "*** Begin Patch\n*** Add File: changed.rs\n+pub fn value() -> u32 { 42 }\n*** End Patch"
                }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text("Updated changed.rs with value(), which returns 42. The check completed successfully.".into())],
            1,
            1,
        ),
    ]
}

fn assert_invalidated(outcome: &TurnOutcome, ui: &LateUi) {
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert_eq!(outcome.stop_reason, TurnStopReason::VerificationUnavailable);
    assert_eq!(outcome.verified_workspace_revision, None);
    assert!(!ui.events.contains(&hi_events::EventKind::RunCompleted));
    assert!(ui.events.contains(&hi_events::EventKind::RunFailed));
}

#[tokio::test]
async fn suggestion_callback_mutation_invalidates_the_terminal_receipt() {
    for verified in [true, false] {
        let workspace = IsolatedWorkspace::new("suggestion-final-evidence");
        let mut cfg = verified_config(&workspace);
        cfg.memory.suggest_next_prompt = true;
        let mut responses = if verified {
            code_responses()
        } else {
            cfg.routing.tool_mode = hi_ai::ToolMode::ChatOnly;
            cfg.gates.verification = VerificationMode::Disabled;
            vec![completion(
                vec![Content::Text("The answer is 42.".into())],
                1,
                1,
            )]
        };
        responses.push(completion(
            vec![Content::Text("Inspect the change.".into())],
            1,
            1,
        ));
        let mut subject = agent(responses, cfg);
        let records = Arc::new(Mutex::new(Records::default()));
        subject.set_session(Box::new(TerminalSession(records.clone())));
        let mut ui = LateUi {
            mutation: Some(workspace.path("late.rs")),
            events: Vec::new(),
            statuses: Vec::new(),
        };

        let outcome = subject
            .run_turn(
                if verified {
                    "update changed.rs"
                } else {
                    "answer briefly"
                },
                &mut ui,
            )
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "{error:#}; statuses={:?}; messages={:?}",
                    ui.statuses,
                    subject
                        .messages()
                        .iter()
                        .map(Message::text)
                        .collect::<Vec<_>>()
                )
            });

        assert_invalidated(&outcome, &ui);
        assert!(outcome.changed_files.contains(&"late.rs".into()));
        assert!(
            ui.mutation.is_none(),
            "suggestion callback must have executed"
        );
        assert_eq!(records.lock().unwrap().outcomes, vec![outcome]);
    }
}

#[cfg(unix)]
fn install_hook(workspace: &IsolatedWorkspace, name: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(workspace.path(".hi/hooks")).unwrap();
    let path = workspace.path(format!(".hi/hooks/{name}"));
    std::fs::write(
        &path,
        "#!/bin/sh\nprintf 'pub fn late_hook() {}\\n' > hook.rs\n",
    )
    .unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    crate::set_workspace_trusted(&workspace.path(""), true).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn post_turn_and_stop_hook_mutations_cannot_keep_a_passed_outcome() {
    for hook in ["post-turn", "stop"] {
        let workspace = IsolatedWorkspace::new("hook-final-evidence");
        install_hook(&workspace, hook);
        let mut subject = agent(code_responses(), verified_config(&workspace));
        let mut ui = LateUi {
            mutation: None,
            events: Vec::new(),
            statuses: Vec::new(),
        };

        let outcome = subject
            .run_turn("update changed.rs", &mut ui)
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "{error:#}; statuses={:?}; messages={:?}",
                    ui.statuses,
                    subject
                        .messages()
                        .iter()
                        .map(Message::text)
                        .collect::<Vec<_>>()
                )
            });

        assert_invalidated(&outcome, &ui);
        assert!(outcome.changed_files.contains(&"hook.rs".into()));
        assert_eq!(
            subject.workspace_controller_status().state,
            hi_workspace::WorkspaceState::Ready
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn hook_added_code_cannot_inherit_read_only_not_applicable_success() {
    let workspace = IsolatedWorkspace::new("hook-after-read-only");
    install_hook(&workspace, "post-turn");
    let mut cfg = verified_config(&workspace);
    cfg.routing.tool_mode = hi_ai::ToolMode::ChatOnly;
    cfg.gates.verification = VerificationMode::Disabled;
    let mut subject = agent(
        vec![completion(
            vec![Content::Text("The answer is 42.".into())],
            1,
            1,
        )],
        cfg,
    );
    let mut ui = LateUi {
        mutation: None,
        events: Vec::new(),
        statuses: Vec::new(),
    };

    let outcome = subject.run_turn("answer briefly", &mut ui).await.unwrap();

    assert_invalidated(&outcome, &ui);
    assert!(outcome.changed_files.contains(&"hook.rs".into()));
}

#[cfg(unix)]
#[tokio::test]
async fn late_hook_revokes_goal_completion_without_persisting_its_recovery_credit() {
    let workspace = IsolatedWorkspace::new("hook-invalid-goal-credit");
    install_hook(&workspace, "stop");
    let mut cfg = verified_config(&workspace);
    cfg.subagents.long_horizon = true;
    let mut subject = agent(code_responses(), cfg);
    let records = Arc::new(Mutex::new(Records::default()));
    subject.set_session(Box::new(TerminalSession(records.clone())));
    let mut goal = Goal::new(
        "ship the change",
        vec!["first change".into(), "follow-up".into()],
    );
    goal.team = false;
    subject.set_structured_goal(Some(goal)).unwrap();
    let mut ui = LateUi {
        mutation: None,
        events: Vec::new(),
        statuses: Vec::new(),
    };

    let outcome = subject
        .run_turn("update changed.rs", &mut ui)
        .await
        .unwrap();

    assert_invalidated(&outcome, &ui);
    let goal = subject.structured_goal().unwrap();
    assert_eq!(goal.active_index(), Some(0));
    assert_eq!(goal.sub_goals[0].status, GoalStatus::Active);
    let records = records.lock().unwrap();
    assert_eq!(records.goals.last(), Some(goal));
    assert!(
        records
            .goals
            .iter()
            .all(|goal| goal.sub_goals[0].status != GoalStatus::Done)
    );
    for state in records
        .recovery
        .iter()
        .chain(std::iter::once(subject.task_recovery()))
    {
        let saved = serde_json::to_value(state).unwrap();
        assert!(
            saved["completed_effects"].as_array().unwrap().is_empty(),
            "unsupported goal credit: {saved}"
        );
    }
}

#[tokio::test]
async fn goal_completion_is_persisted_only_with_the_final_applicable_receipt() {
    for fail_receipt in [false, true] {
        let workspace = IsolatedWorkspace::new("goal-credit-atomic-runtime");
        let mut cfg = verified_config(&workspace);
        cfg.subagents.long_horizon = true;
        let mut subject = agent(code_responses(), cfg);
        let records = Arc::new(Mutex::new(Records {
            fail_receipt,
            ..Records::default()
        }));
        subject.set_session(Box::new(TerminalSession(records.clone())));
        let mut goal = Goal::new(
            "ship the change",
            vec!["first change".into(), "follow-up".into()],
        );
        goal.team = false;
        subject.set_structured_goal(Some(goal)).unwrap();

        let mut ui = RecUi::default();
        let result = subject.run_turn("update changed.rs", &mut ui).await;

        if fail_receipt {
            let error = result.unwrap_err();
            assert!(format!("{error:#}").contains("terminal receipt unavailable"));
            assert_eq!(subject.structured_goal().unwrap().active_index(), Some(0));
            let records = records.lock().unwrap();
            assert!(records.outcomes.is_empty());
            assert!(
                records
                    .goals
                    .iter()
                    .all(|goal| goal.sub_goals[0].status != GoalStatus::Done)
            );
            continue;
        }
        let outcome = result.unwrap();

        assert_eq!(outcome.status, TurnStatus::Completed);
        let goal = subject.structured_goal().unwrap();
        assert_eq!(
            goal.sub_goals[0].status,
            GoalStatus::Done,
            "statuses={:?}; goal={goal:?}; outcome={outcome:?}; verify={:?}; recovery={:?}",
            ui.statuses,
            subject.report.verify,
            subject.task_recovery()
        );
        let records = records.lock().unwrap();
        assert_eq!(records.outcomes, vec![outcome]);
        assert_eq!(records.goals.last(), Some(goal));
        assert!(
            records.goals[..records.goals.len() - 1]
                .iter()
                .all(|goal| goal.sub_goals[0].status != GoalStatus::Done)
        );
        let saved = serde_json::to_value(subject.task_recovery()).unwrap();
        assert!(
            saved["completed_effects"]
                .as_array()
                .unwrap()
                .iter()
                .any(|effect| effect.as_str() == Some("goal:ship the change:0"))
        );
    }
}
