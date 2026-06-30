use super::*;
use super::common::*;

#[test]
fn goal_updates_system_prompt_and_clear_history_keeps_it() {
    let mut agent = agent(vec![], config());
    agent.set_goal(Some("ship a stable TUI".into()));

    assert_eq!(agent.goal(), Some("ship a stable TUI"));
    assert!(
        agent.messages()[0]
            .text()
            .contains("[Current session goal]"),
        "goal marker included"
    );
    assert!(
        agent.messages()[0].text().contains("ship a stable TUI"),
        "goal text included"
    );

    agent.messages_mut().push(Message::user("noise"));
    agent.clear_history();
    assert_eq!(agent.messages().len(), 1);
    assert!(
        agent.messages()[0].text().contains("ship a stable TUI"),
        "goal survives clear-history"
    );

    agent.set_goal(None);
    assert_eq!(agent.goal(), None);
    assert!(
        !agent.messages()[0]
            .text()
            .contains("[Current session goal]"),
        "goal marker removed"
    );
}


#[tokio::test]
async fn structured_goal_state_injected_into_system_prompt_when_long_horizon_on() {
    // With long_horizon on, a structured goal's state (objective + sub-goal
    // checklist + retry notes) is injected into the system prompt so the
    // agent resumes the active sub-goal coherently each turn.
    let mut cfg = config();
    cfg.long_horizon = true;
    let mut agent = agent(
        vec![completion(vec![Content::Text("ok".into())], 1, 1)],
        cfg,
    );
    let mut goal = Goal::new(
        "refactor the parser",
        vec!["write tests".into(), "rewrite parser".into()],
    );
    // Record a failed attempt so the prompt surfaces "don't repeat" notes.
    goal.record_failure("approach A didn't compile", DEFAULT_SUBGOAL_RETRIES);
    assert!(
        agent.set_structured_goal(Some(goal)),
        "accepted when long_horizon on"
    );

    let sys = agent.messages()[0].text();
    assert!(sys.contains("Long-horizon goal"), "header: {sys}");
    assert!(sys.contains("refactor the parser"), "objective: {sys}");
    assert!(sys.contains("write tests"), "sub-goal: {sys}");
    assert!(
        sys.contains("don't repeat these"),
        "retry notes surfaced: {sys}"
    );

    // Clearing the goal removes the section.
    agent.set_structured_goal(None);
    let sys_after = agent.messages()[0].text();
    assert!(
        !sys_after.contains("Long-horizon goal"),
        "goal section cleared: {sys_after}"
    );
}

#[tokio::test]
async fn structured_goal_rejected_when_long_horizon_off() {
    // Default config has long_horizon off — setting a structured goal is
    // rejected (the single-turn loop is unchanged), so the system prompt
    // gains no goal section.
    let mut agent = agent(
        vec![completion(vec![Content::Text("ok".into())], 1, 1)],
        config(),
    );
    let goal = Goal::new("do a thing", vec!["step one".into()]);
    assert!(!agent.set_structured_goal(Some(goal)), "rejected when off");
    assert!(agent.structured_goal().is_none());
    let sys = agent.messages()[0].text();
    assert!(
        !sys.contains("Long-horizon goal"),
        "no goal section when off: {sys}"
    );
}

#[tokio::test]
async fn long_horizon_driver_advances_on_clean_turn() {
    // With long_horizon on and a structured goal set, a turn that verifies
    // clean (or has no verify and doesn't stall) advances the active
    // sub-goal, and the system prompt reflects the new active sub-goal.
    let mut cfg = config();
    cfg.long_horizon = true;
    // One turn: model writes a file (tool), then a clean text finish. No
    // verify configured → a non-stalling turn with no verify is "clean".
    let tmp = temp_file("lh1");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    agent.set_structured_goal(Some(Goal::new(
        "refactor",
        vec!["step one".into(), "step two".into()],
    )));
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();
    let _ = std::fs::remove_file(&tmp);
    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.sub_goals[0].status,
        GoalStatus::Done,
        "advanced past step 1"
    );
    assert_eq!(goal.active_index(), Some(1), "step 2 now active");
    // The system prompt reflects the new active sub-goal.
    assert!(
        agent.messages()[0].text().contains("step two"),
        "system prompt shows new active sub-goal"
    );
}

#[tokio::test]
async fn long_horizon_driver_records_failure_on_stall() {
    // A turn that stalls (repeat guard exhausts) records a sub-goal attempt
    // so the next turn sees the prior note (and doesn't repeat the approach).
    let mut cfg = config();
    cfg.long_horizon = true;
    cfg.max_repeat_nudges = 1;
    // Model re-issues the same tool call → repeat guard stalls the turn
    // after exhausting the (1) nudge budget. Three identical writes: the
    // second triggers a nudge, the third exhausts the budget and breaks
    // stalled.
    let responses = vec![
        write_completion("lhstall"),
        write_completion("lhstall"),
        write_completion("lhstall"),
    ];
    let mut agent = agent(responses, cfg);
    agent.set_structured_goal(Some(Goal::new(
        "refactor",
        vec!["step one".into(), "step two".into()],
    )));
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();
    let _ = std::fs::remove_file("lhstall");
    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(goal.active_index(), Some(0), "didn't advance (stalled)");
    assert!(
        goal.sub_goals[0].attempts > 0,
        "recorded a failure attempt: {:?}",
        goal.sub_goals[0]
    );
    assert!(
        goal.sub_goals[0]
            .notes
            .iter()
            .any(|n| n.contains("stalled")),
        "stall reason recorded as a note: {:?}",
        goal.sub_goals[0].notes
    );
    // The system prompt surfaces the "don't repeat" notes on the active
    // sub-goal, so the next turn doesn't repeat the failed approach.
    assert!(
        agent.messages()[0].text().contains("don't repeat these"),
        "retry notes in system prompt"
    );
}

