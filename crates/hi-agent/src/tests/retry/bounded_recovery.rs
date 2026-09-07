use super::*;

#[tokio::test]
async fn synthetic_drive_context_loss_cannot_reset_restored_exhaustion() {
    for (prompt, goal_done) in [
        (crate::GOAL_CONTINUE_PROMPT.to_owned(), false),
        (format!("{}\n", crate::GOAL_CONTINUE_PROMPT), false),
        (crate::GOAL_CONTINUE_PROMPT.to_owned(), true),
        (crate::PLAN_DRIVE_PROMPT.to_owned(), false),
        (format!("{}\n", crate::PLAN_DRIVE_PROMPT), false),
    ] {
        let workspace = IsolatedWorkspace::new("exhausted-synthetic-drive");
        let mut cfg = workspace.config();
        cfg.subagents.long_horizon = true;
        cfg.gates.verification = VerificationMode::Disabled;
        cfg.memory.finalize = false;
        cfg.memory.suggest_next_prompt = false;
        let (mut subject, requests) = scripted_agent(vec![], cfg);
        let mut goal = Goal::new("original objective", vec!["implement the change".into()]);
        goal.team = false;
        if goal_done {
            goal.advance();
        }
        subject.set_structured_goal(Some(goal)).unwrap();
        if crate::DriveKind::from_prompt(&prompt) == crate::DriveKind::Goal && !goal_done {
            assert_eq!(
                subject.goal_continuation_context(&prompt),
                subject.goal_continuation_context(crate::GOAL_CONTINUE_PROMPT),
            );
        }
        let mut recovery = crate::TaskRecoveryState::new("original objective".into(), 3);
        recovery.stop("prior automatic recovery exhausted");
        let restored = serde_json::from_str(&serde_json::to_string(&recovery).unwrap()).unwrap();
        subject.restore_task_recovery(restored);

        let outcome = subject.run_turn(&prompt, &mut NullUi).await.unwrap();

        assert!(
            requests.lock().unwrap().is_empty(),
            "synthetic prompt {prompt:?}"
        );
        assert_eq!(outcome.status, TurnStatus::Failed);
        assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
        assert!(subject.task_recovery().exhausted);
        assert_eq!(subject.task_recovery().remaining, 0);
        assert_eq!(subject.task_recovery().objective, "original objective");
    }
}

#[tokio::test]
async fn real_user_turn_starts_a_new_allowance_after_exhaustion() {
    let workspace = IsolatedWorkspace::new("new-user-task-recovery");
    let mut cfg = workspace.config();
    cfg.routing.tool_mode = hi_ai::ToolMode::ChatOnly;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let (mut subject, requests) = scripted_agent(
        vec![ProviderStep::Completion(completion(
            vec![Content::Text("Hello.".into())],
            1,
            1,
        ))],
        cfg,
    );
    let mut previous = crate::TaskRecoveryState::new("previous task".into(), 3);
    previous.stop("automatic recovery exhausted");
    subject.restore_task_recovery(previous);
    subject
        .run_turn("Say hello in one word.", &mut NullUi)
        .await
        .unwrap();
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert!(!subject.task_recovery().exhausted);
    assert_eq!(
        subject.task_recovery().remaining,
        subject.task_recovery().limit
    );
    assert_eq!(subject.task_recovery().objective, "Say hello in one word.");
}

#[tokio::test]
async fn alternating_invalid_tool_turns_share_the_task_recovery_allowance() {
    // A model that alternates a valid tool call with an invalid tool turn keeps
    // resetting the *consecutive* protocol counter (MAX_TOOL_PROTOCOL_RETRIES), so
    // without the cumulative cap the nudge-and-retry loop runs forever (the qtest4
    // wedge). The cumulative circuit-breaker must end the turn instead. Distinct
    // valid calls each round keep the repeat-tool-call guard from firing first, so
    // this isolates the protocol cap; far more pairs than the cap are scripted, so
    // a non-terminating loop would exhaust the script and panic in the provider.
    let mut steps = Vec::new();
    for i in 0..16 {
        steps.push(ProviderStep::Completion(bash_completion(&format!(
            "echo {i}"
        ))));
        steps.push(ProviderStep::Error(ProviderErrorKind::ToolProtocol));
    }
    let (mut agent, requests) = scripted_agent(steps, config());
    let mut ui = RecUi::default();

    let outcome = agent.run_turn("go", &mut ui).await.unwrap();
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert!(agent.task_recovery().exhausted);
    assert!(
        requests.lock().unwrap().len() < 16,
        "alternating successes must not buy unlimited corrections"
    );

    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("automatic recovery exhausted")),
        "the shared allowance should end alternating invalid turns: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn implementation_tool_protocol_exhaustion_falls_back_to_text_tool_calls() {
    let path = temp_file("protocol-text-fallback");
    let path_string = path.to_string_lossy().to_string();
    let xmlish_write = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>{path_string}</arg_value><arg_key>content</arg_key><arg_value>ok\n</arg_value></tool_call>"
    );
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Completion(completion(vec![Content::Text(xmlish_write)], 5, 3)),
            ProviderStep::Completion(bash_completion("true # validate")),
            ProviderStep::Completion(completion(
                vec![Content::Text(format!(
                    "Changed {path_string} and validated with true # validate."
                ))],
                5,
                3,
            )),
        ],
        config(),
    );
    let mut ui = RecordingUi::default();
    agent
        .run_turn("/build a small CLI project tracker", &mut ui)
        .await
        .unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "ok\n");
    let _ = std::fs::remove_file(&path);

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("plain-text tool-call parsing")),
        "expected text-tool fallback status: {:?}",
        ui.statuses
    );
    assert!(
        agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("validated with true # validate")
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        6,
        "three invalid sends, one safe text-tool call, validation, and final answer"
    );
}
