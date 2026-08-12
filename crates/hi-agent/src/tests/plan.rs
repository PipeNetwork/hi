use super::common::*;
use super::*;

#[tokio::test]
async fn bounded_discovery_plan_transitions_to_verified_mutation() {
    let workspace = IsolatedWorkspace::new("mixed-review-build");
    let mut responses = Vec::new();
    // Reproduce the failed live turn: ten reads, a nudge, two more reads,
    // another nudge, one more read, then a concrete plan. Every read must
    // remain available and execute successfully throughout recovery.
    for index in 0..13 {
        let relative = format!("src/context-{index}.rs");
        let path = workspace.path(&relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            format!("pub const VALUE_{index}: usize = {index};\n"),
        )
        .unwrap();
        responses.push(completion(
            vec![Content::ToolCall {
                id: format!("read-{index}"),
                name: "read".into(),
                arguments: serde_json::json!({"path": relative}).to_string(),
            }],
            1,
            1,
        ));
    }
    responses.push(completion(
        vec![Content::ToolCall {
            id: "plan-active".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "Implement the selected component", "status": "active"}]
            })
            .to_string(),
        }],
        1,
        1,
    ));
    let post_plan_read = workspace.path("src/post-plan-context.rs");
    std::fs::write(&post_plan_read, "final context before the edit\n").unwrap();
    responses.push(completion(
        vec![Content::ToolCall {
            id: "post-plan-read".into(),
            name: "read".into(),
            arguments: serde_json::json!({"path": "src/post-plan-context.rs"}).to_string(),
        }],
        1,
        1,
    ));
    let changed = workspace.path("src/implemented.rs");
    responses.push(write_completion(&changed.to_string_lossy()));
    responses.push(completion(
        vec![Content::ToolCall {
            id: "plan-done".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "Implement the selected component", "status": "done"}]
            })
            .to_string(),
        }],
        1,
        1,
    ));
    responses.push(bash_completion("true # validate"));
    responses.push(completion(vec![Content::Text("implemented".into())], 1, 1));

    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(responses),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
    };
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("review plan.md and lets keep building this", &mut ui)
        .await
        .unwrap();

    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "outcome={outcome:?}; statuses={:?}",
        ui.statuses
    );
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        outcome
            .changed_files
            .iter()
            .any(|path| path == "src/implemented.rs")
    );
    assert!(outcome.verified_workspace_revision.is_some());
    assert!(changed.exists());
    assert!(
        ui.statuses.iter().any(|status| status.contains(
            "mutation request used 10 model rounds (10 tools) without editing; requesting an implementation step"
        )),
        "discovery loop should be bounded: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("plan recorded after bounded discovery")),
        "the plan-to-edit transition must be visible: {:?}",
        ui.statuses
    );
    assert!(!agent.last_turn_telemetry().stalled_unfinished);
    let read_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "read")
        .collect::<Vec<_>>();
    assert_eq!(read_results.len(), 14, "all scripted reads must execute");
    assert!(
        read_results
            .iter()
            .all(|(_, result)| !result.to_ascii_lowercase().contains("denied")),
        "discovery steering must never deny read: {read_results:?}"
    );
    assert!(
        agent
            .last_turn_telemetry()
            .tool_timeline
            .iter()
            .filter(|entry| entry.tool == "read")
            .all(|entry| entry.status == hi_tools::ToolStatus::Succeeded),
        "every read must have typed Succeeded status"
    );
    let guided_tools = tool_names.lock().unwrap()[10]
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert!(guided_tools.contains("read"));
    assert!(guided_tools.contains("update_plan"));
    assert!(guided_tools.contains("write"));
    assert_eq!(modes.lock().unwrap()[10], ToolMode::Required);
    assert_eq!(modes.lock().unwrap()[12], ToolMode::Required);
    let post_plan_tools = tool_names.lock().unwrap()[14]
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert!(post_plan_tools.contains("read"));
    assert!(post_plan_tools.contains("write"));
    assert_eq!(modes.lock().unwrap()[14], ToolMode::Required);
    assert_eq!(modes.lock().unwrap()[15], ToolMode::Required);
}

#[tokio::test]
async fn resumed_active_plan_transitions_to_mutation_instead_of_stalling() {
    let workspace = IsolatedWorkspace::new("resumed-plan-build");
    let mut responses = Vec::new();
    for index in 0..10 {
        let relative = format!("src/resumed-context-{index}.rs");
        std::fs::create_dir_all(workspace.path("src")).unwrap();
        std::fs::write(workspace.path(&relative), format!("context {index}\n")).unwrap();
        responses.push(completion(
            vec![Content::ToolCall {
                id: format!("resumed-read-{index}"),
                name: "read".into(),
                arguments: serde_json::json!({"path": relative}).to_string(),
            }],
            1,
            1,
        ));
    }
    let changed = workspace.path("src/resumed-implemented.rs");
    responses.push(write_completion(&changed.to_string_lossy()));
    responses.push(completion(
        vec![Content::ToolCall {
            id: "resumed-plan-done".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "Resume implementation", "status": "done"}]
            })
            .to_string(),
        }],
        1,
        1,
    ));
    responses.push(bash_completion("true # validate"));
    responses.push(completion(vec![Content::Text("implemented".into())], 1, 1));

    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(responses),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
    };
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    agent.goals.last_plan = vec![PlanStep {
        title: "Resume implementation".into(),
        status: PlanStatus::Active,
    }];
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("continue building this", &mut ui)
        .await
        .unwrap();

    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "outcome={outcome:?}; statuses={:?}",
        ui.statuses
    );
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert!(!agent.last_turn_telemetry().stalled_unfinished);
    assert!(changed.exists());
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("active implementation plan already exists"))
    );
    let recovery_tools = tool_names.lock().unwrap()[10]
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert!(recovery_tools.contains("read"));
    assert!(recovery_tools.contains("write"));
    assert!(
        agent
            .last_turn_telemetry()
            .tool_timeline
            .iter()
            .filter(|entry| entry.tool == "read")
            .all(|entry| entry.status == hi_tools::ToolStatus::Succeeded)
    );
    assert_eq!(modes.lock().unwrap()[10], ToolMode::Required);
}

#[tokio::test]
async fn plan_with_pending_steps_continues_past_recap() {
    // The model posts a plan (2/3 done), does one step, then stops with a
    // finished-looking recap. Without plan-awareness, the text heuristic
    // sees a finished recap and ends the turn — leaving the plan at 2/3.
    // With plan-awareness, the agent detects pending steps and nudges the
    // model to continue until the plan is complete.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 5;
    // Helper: an update_plan call with given step statuses.
    let plan_call = |id: &str, statuses: &[&str]| {
        let steps: Vec<String> = statuses
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#"{{"title":"step {}","status":"{}"}}"#, i + 1, s))
            .collect();
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // R1: model posts the initial plan (0/3 done) and starts step 1.
        plan_call("p1", &["active", "pending", "pending"]),
        // R2: model does a read for step 1.
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R3: model updates plan (1/3 done, step 2 active) and does a read.
        plan_call("p2", &["done", "active", "pending"]),
        // R4: model stops with a finished-looking recap — but plan is 1/3!
        // The plan-aware continue should nudge it to keep going.
        completion(
            vec![Content::Text(
                "I've completed step 1. The implementation looks good.".into(),
            )],
            1,
            1,
        ),
        // R5 (nudged): model does step 2.
        completion(
            vec![Content::ToolCall {
                id: "r2".into(),
                name: "read".into(),
                arguments: r#"{"path":"y"}"#.into(),
            }],
            1,
            1,
        ),
        // R6: model updates plan (2/3 done, step 3 active).
        plan_call("p3", &["done", "done", "active"]),
        // R7: model stops with recap again — plan is 2/3, nudge again.
        completion(
            vec![Content::Text("Step 2 is done. Moving on.".into())],
            1,
            1,
        ),
        // R8 (nudged): model does step 3.
        completion(
            vec![Content::ToolCall {
                id: "r3".into(),
                name: "read".into(),
                arguments: r#"{"path":"z"}"#.into(),
            }],
            1,
            1,
        ),
        // R9: model updates plan (3/3 done) — all complete.
        plan_call("p4", &["done", "done", "done"]),
        // R10: model gives final recap — plan is complete, turn ends.
        completion(
            vec![Content::Text("All steps complete. Done.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        .run_turn("implement the feature", &mut ui)
        .await
        .unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    // The turn should have run all the way to the final recap (R10),
    // not stopped at R4 or R7 when the model gave a partial recap.
    assert!(
        agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("All steps complete"),
        "turn ran to the final recap with plan complete: {:?}",
        agent.messages().last().unwrap().text()
    );
}

#[tokio::test]
async fn new_task_keeps_incomplete_plan_and_folds_replace_nudge() {
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("new task done".into())],
            1,
            1,
        )],
        config(),
    );
    agent.goals.last_plan = vec![PlanStep {
        title: "old unfinished step".into(),
        status: PlanStatus::Pending,
    }];
    let mut ui = RecUi::default();

    agent
        .run_turn("do a different task", &mut ui)
        .await
        .unwrap();

    assert_eq!(agent.goals.last_plan.len(), 1);
    assert_eq!(agent.goals.last_plan[0].title, "old unfinished step");
    assert!(
        ui.plans.is_empty(),
        "incomplete plans must stay pinned: {:?}",
        ui.plans
    );
    let prompt = agent
        .messages()
        .iter()
        .find(|m| m.role == hi_ai::Role::User)
        .map(|m| m.text())
        .unwrap_or_default();
    assert!(
        prompt.contains("If this message is a new task"),
        "replace-plan nudge should fold into the new-task prompt: {prompt}"
    );
}

#[tokio::test]
async fn continue_does_not_preserve_a_completed_plan_box() {
    let mut agent = agent(
        vec![completion(vec![Content::Text("done".into())], 1, 1)],
        config(),
    );
    agent.goals.last_plan = vec![PlanStep {
        title: "old completed step".into(),
        status: PlanStatus::Done,
    }];
    let mut ui = RecUi::default();

    agent.run_turn("continue", &mut ui).await.unwrap();

    assert!(agent.goals.last_plan.is_empty());
    assert_eq!(ui.plans, vec![Vec::<PlanStep>::new()]);
}

#[tokio::test]
async fn complete_plan_ends_turn_without_spurious_continue() {
    // When the plan is fully done (all steps "done"), the model's recap
    // should end the turn cleanly — no plan-driven continue nudge.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 5;
    let plan_call = |id: &str, statuses: &[&str]| {
        let steps: Vec<String> = statuses
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#"{{"title":"step {}","status":"{}"}}"#, i + 1, s))
            .collect();
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // Model posts plan (all done) and gives final recap.
        plan_call("p1", &["done", "done"]),
        completion(vec![Content::Text("All done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    // No spurious continue — the turn ended after exactly 2 responses.
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "no incomplete warning when plan is done: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn long_plan_10_steps_runs_to_completion() {
    // A 10-step plan where the model does one step per round, then stops
    // with a recap. The plan-aware continue should nudge it to keep going
    // until all 10 steps are done. The silent_continues counter resets on
    // each tool call, so this should work regardless of plan length.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 3; // the default
    // The dynamic catalog omits coordination tools for ordinary read-only
    // requests; this fixture specifically exercises update_plan mechanics.
    cfg.memory.tool_set = ToolSet::Full;
    let n_steps = 10;
    let plan_call = |id: &str, statuses: &[&str]| {
        let steps: Vec<String> = statuses
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#"{{"title":"step {}","status":"{}"}}"#, i + 1, s))
            .collect();
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
            }],
            1,
            1,
        )
    };
    let read_call = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        )
    };
    let recap = |text: &str| completion(vec![Content::Text(text.into())], 1, 1);

    let mut responses = Vec::new();
    for step in 0..n_steps {
        // Build statuses: steps before `step` are done, step `step` is active,
        // steps after are pending.
        let statuses: Vec<&str> = (0..n_steps)
            .map(|i| {
                if i < step {
                    "done"
                } else if i == step {
                    "active"
                } else {
                    "pending"
                }
            })
            .collect();
        // Model posts plan + does a read for this step.
        responses.push(plan_call(&format!("p{step}"), &statuses));
        responses.push(read_call(&format!("r{step}")));
        // Model stops with a recap (unless it's the last step).
        if step < n_steps - 1 {
            responses.push(recap(&format!(
                "Step {} is done. The implementation looks good.",
                step + 1
            )));
        }
    }
    // Final: all steps done + final recap.
    let all_done: Vec<&str> = (0..n_steps).map(|_| "done").collect();
    responses.push(plan_call("pfinal", &all_done));
    responses.push(recap("All 10 steps complete. Done."));

    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        // This fixture exercises plan continuation with inspection-only tool
        // calls. Keep the request explicitly read-only so the mutation
        // contract does not correctly stop it after bounded discovery.
        .run_turn("review the feature plan", &mut ui)
        .await
        .unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    // The turn should have run all the way to the final recap.
    let last_text = agent.messages().last().unwrap().text();
    assert!(
        last_text.contains("All 10 steps complete"),
        "turn ran to the final recap, got: {last_text}; statuses: {:?}",
        ui.statuses,
    );
    // Should NOT have ended with an incomplete warning.
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "no incomplete warning on a completed 10-step plan: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn long_plan_survives_text_only_response_to_nudge() {
    // A plan where the model sometimes responds to the continue-nudge with
    // text-only (no tool call) before eventually doing the work. This is
    // the real-world pattern that causes stalls: the model writes a recap,
    // gets nudged, writes another recap instead of acting, gets nudged
    // again, and eventually does the work. The silent_continues budget
    // must be high enough to survive a few text-only responses.
    //
    // With max_silent_continues=3, the model can text-only 3 times in a
    // row before the turn ends. On the 4th text-only, the budget is
    // exhausted. This test has 3 text-only responses (within budget)
    // before the model finally acts.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 3;
    let plan_call = |id: &str, s1: &str, s2: &str, s3: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"a","status":"{s1}"}},{{"title":"b","status":"{s2}"}},{{"title":"c","status":"{s3}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // R1: plan + read for step 1.
        plan_call("p1", "active", "pending", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R2: recap, no tools → nudge (silent_continues=1, force_tools).
        completion(vec![Content::Text("Step 1 done. Looks good.".into())], 1, 1),
        // R3: text-only again (ignores force) → nudge (silent_continues=2).
        completion(
            vec![Content::Text(
                "The implementation is clean. No issues found.".into(),
            )],
            1,
            1,
        ),
        // R4: text-only again (ignores force) → nudge (silent_continues=3).
        completion(
            vec![Content::Text("Everything looks correct so far.".into())],
            1,
            1,
        ),
        // R5: finally does a tool call → silent_continues resets to 0.
        plan_call("p2", "done", "active", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r2".into(),
                name: "read".into(),
                arguments: r#"{"path":"y"}"#.into(),
            }],
            1,
            1,
        ),
        // R6: recap → nudge (silent_continues=1).
        completion(vec![Content::Text("Step 2 done.".into())], 1, 1),
        // R7: does step 3.
        plan_call("p3", "done", "done", "active"),
        completion(
            vec![Content::ToolCall {
                id: "r3".into(),
                name: "read".into(),
                arguments: r#"{"path":"z"}"#.into(),
            }],
            1,
            1,
        ),
        // R8: all done + final recap.
        plan_call("p4", "done", "done", "done"),
        completion(
            vec![Content::Text("All steps complete. Done.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    let last_text = agent.messages().last().unwrap().text();
    assert!(
        last_text.contains("All steps complete"),
        "turn ran to completion despite text-only responses to nudges, got: {last_text}"
    );
}

#[tokio::test]
async fn plan_stalls_after_max_consecutive_text_only_responses() {
    // When the model responds to the continue-nudge with text-only (no tool
    // call) more than max_silent_continues times in a row, the turn ends
    // with an "incomplete" warning. This is the safety valve — the model is
    // stuck narrating without acting. This test verifies the valve fires
    // at the right point: after exactly max_silent_continues+1 text-only
    // responses (the original recap + max_silent_continues nudged retries).
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 3;
    let plan_call = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: r#"{"steps":[{"title":"a","status":"active"},{"title":"b","status":"pending"}]}"#.into(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // R1: plan + read for step 1.
        plan_call("p1"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R2: recap → nudge (1/3).
        completion(vec![Content::Text("Step 1 done.".into())], 1, 1),
        // R3: text-only → nudge (2/3).
        completion(vec![Content::Text("Looks good.".into())], 1, 1),
        // R4: text-only → nudge (3/3).
        completion(vec![Content::Text("Correct.".into())], 1, 1),
        // R5: text-only → budget exhausted, turn ends with warning.
        completion(vec![Content::Text("Fine.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn ended");
    // Should warn about incomplete — the model kept narrating without acting.
    assert!(
        ui.statuses.iter().any(|s| s.contains("incomplete")),
        "should warn incomplete after exhausting continue budget: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn keep_working_finishes_plan_after_silent_continue_budget() {
    // After the silent-continue budget is spent, production keeps working
    // in-turn instead of asking the user to retry. The model then acts and
    // finishes the plan.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 1;
    cfg.loop_limits.max_keep_working = 2;
    let plan_call = |id: &str, s1: &str, s2: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"a","status":"{s1}"}},{{"title":"b","status":"{s2}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        plan_call("p1", "active", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        completion(vec![Content::Text("Step 1 done.".into())], 1, 1),
        completion(vec![Content::Text("Looks good.".into())], 1, 1),
        // Silent-continue budget spent; keep-working must fire here.
        completion(vec![Content::Text("Still recapping.".into())], 1, 1),
        plan_call("p2", "done", "done"),
        completion(
            vec![Content::Text("All steps complete. Done.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(
        ui.statuses.iter().any(|s| s.contains("still working")),
        "keep-working recovery should fire after silent-continue budget: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("/retry")),
        "must not ask the user to retry: {:?}",
        ui.statuses
    );
    let last_text = agent.messages().last().unwrap().text();
    assert!(
        last_text.contains("All steps complete"),
        "turn should finish the plan, got: {last_text}"
    );
}

#[tokio::test]
async fn plan_persists_across_turns_for_continue() {
    // When a turn ends with an incomplete plan and the user types
    // "continue", the plan state should persist so the plan-aware continue
    // logic can fire. Without persistence, last_plan is cleared at the
    // start of the new turn and the agent can't detect the incomplete plan.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 3;
    let plan_call = |id: &str, s1: &str, s2: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"a","status":"{s1}"}},{{"title":"b","status":"{s2}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };

    // Turn 1: model posts plan (step 1 active), does step 1, then stops
    // with a recap. The plan-continue nudges, but the model text-only's
    // past the budget, so the turn ends with an incomplete plan (1/2).
    let turn1_responses = vec![
        plan_call("p1", "active", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // Recap → nudge (1/3).
        completion(vec![Content::Text("Step 1 done.".into())], 1, 1),
        // Text-only → nudge (2/3).
        completion(vec![Content::Text("Looks good.".into())], 1, 1),
        // Text-only → nudge (3/3).
        completion(vec![Content::Text("Correct.".into())], 1, 1),
        // Text-only → budget exhausted, turn ends.
        completion(vec![Content::Text("Fine.".into())], 1, 1),
    ];
    let mut agent = agent(turn1_responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    // Turn 1 ended with incomplete warning — plan is 1/2.
    assert!(
        ui.statuses.iter().any(|s| s.contains("incomplete")),
        "turn 1 should end incomplete: {:?}",
        ui.statuses
    );

    // Verify the plan state persisted after turn 1 — it should still have
    // pending steps so the plan-aware continue can fire on "continue".
    let plan_after_turn1 = &agent.goals.last_plan;
    assert!(
        plan_has_pending_steps(plan_after_turn1),
        "plan should persist with pending steps after turn 1: {:?}",
        plan_after_turn1
    );

    // Turn 2: user types "fix a different bug" (NOT "continue"). The plan
    // stays pinned; a replace-plan nudge tells the model to swap the
    // checklist if this is a new task.
    assert!(
        !looks_like_continue("fix a different bug"),
        "a new task should not look like continue"
    );
    assert!(
        looks_like_continue("continue"),
        "'continue' should look like continue"
    );
    assert!(
        plan_has_pending_steps(&agent.goals.last_plan),
        "incomplete plan must survive a new-task follow-up: {:?}",
        agent.goals.last_plan
    );
}

fn pending_step(title: &str) -> PlanStep {
    PlanStep {
        title: title.into(),
        status: PlanStatus::Pending,
    }
}

fn completed_outcome(leftover: Option<String>) -> TurnOutcome {
    TurnOutcome {
        status: TurnStatus::Completed,
        verification: VerificationStatus::Unverified,
        review: ReviewStatus::NotRequired,
        stop_reason: TurnStopReason::Completed,
        changed_files: Vec::new(),
        verified_workspace_revision: None,
        effective_route: crate::EffectiveModelRoute {
            provider: Some("test".into()),
            model: "m".into(),
        },
        review_same_model: false,
        leftover: leftover.clone(),
        plan_leftover: leftover,
    }
}

#[tokio::test]
async fn plan_drive_prompt_finishes_next_pending_step() {
    let workspace = IsolatedWorkspace::new("plan-drive-finish");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_silent_continues = 0;
    cfg.loop_limits.max_keep_working = 0;
    let plan_call = |id: &str, s1: &str, s2: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"a","status":"{s1}"}},{{"title":"b","status":"{s2}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        plan_call("p1", "done", "pending"),
        write_completion(&workspace.path("src/a.rs").to_string_lossy()),
        completion(vec![Content::Text("Step a is done.".into())], 1, 1),
        plan_call("p2", "done", "done"),
        completion(vec![Content::Text("All steps complete.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    let first = agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(
        agent.plan_incomplete(),
        "first turn should leave step b pending: {:?}",
        agent.current_plan()
    );
    assert!(
        agent.drive_decision(Some(&first)).should_enqueue(),
        "leftover plan should auto-drive after the first turn: {first:?}"
    );

    let second = agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut ui)
        .await
        .unwrap();
    assert!(
        !agent.plan_incomplete(),
        "plan-drive should finish the remaining step: {:?}",
        agent.current_plan()
    );
    assert!(!agent.drive_decision(Some(&second)).should_enqueue());
}

#[test]
fn completed_turn_with_pending_plan_still_auto_drives() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    let leftover = agent.leftover_work();
    assert_eq!(
        leftover.as_deref(),
        Some("1/1 remaining — wire the scheduler")
    );
    let outcome = completed_outcome(leftover.clone());
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Enqueue
    );
    agent.set_plan_mode(true);
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::PlanMode
        }
    );
    agent.set_plan_mode(false);
    agent.set_plan_drive_paused(true);
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::Paused
        }
    );
}

#[test]
fn four_no_progress_plan_drives_park_and_mutation_resets() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    let outcome = completed_outcome(agent.leftover_work());
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Enqueue
    );
    for _ in 0..4 {
        agent.note_plan_drive_progress(false);
    }
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::Parked
        }
    );
    assert_eq!(agent.plan_drive_status(), "parked");
    agent.note_plan_drive_progress(true);
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Enqueue
    );
}

#[tokio::test]
async fn completed_turn_stamps_leftover_when_plan_pending() {
    let workspace = IsolatedWorkspace::new("completed-leftover");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_silent_continues = 0;
    cfg.loop_limits.max_keep_working = 0;
    let mut agent = agent(
        vec![
            completion(
                vec![Content::ToolCall {
                    id: "p1".into(),
                    name: "update_plan".into(),
                    arguments: r#"{"steps":[{"title":"a","status":"done"},{"title":"b","status":"pending"}]}"#.into(),
                }],
                1,
                1,
            ),
            write_completion(&workspace.path("src/a.rs").to_string_lossy()),
            completion(vec![Content::Text("Step a is done.".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("do it", &mut ui).await.unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(
        outcome
            .leftover
            .as_deref()
            .is_some_and(|line| line.contains("remaining")),
        "Completed leftover should name remaining work: {outcome:?}"
    );
    assert!(agent.drive_decision(Some(&outcome)).should_enqueue());
}

#[tokio::test]
async fn new_task_without_update_plan_keeps_stale_pin() {
    let workspace = IsolatedWorkspace::new("stale-pin");
    let mut cfg = workspace.config();
    cfg.loop_limits.max_silent_continues = 0;
    cfg.loop_limits.max_keep_working = 0;
    let plan_call = completion(
        vec![Content::ToolCall {
            id: "p1".into(),
            name: "update_plan".into(),
            arguments: r#"{"steps":[{"title":"old work","status":"pending"}]}"#.into(),
        }],
        1,
        1,
    );
    let mut agent = agent(
        vec![
            plan_call,
            completion(vec![Content::Text("planned.".into())], 1, 1),
            completion(
                vec![Content::Text("starting the new auth work.".into())],
                1,
                1,
            ),
            completion(
                vec![Content::Text("still on the old checklist.".into())],
                1,
                1,
            ),
            completion(vec![Content::Text("steering noted.".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(agent.plan_incomplete());

    agent
        .run_turn(
            "implement a new auth system from scratch that replaces the old one",
            &mut ui,
        )
        .await
        .unwrap();
    assert!(
        agent.plan_incomplete(),
        "new-task without update_plan should keep the pin: {:?}",
        agent.current_plan()
    );
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| message.text().contains("/plan replace")),
        "new-task fold should name /plan replace"
    );

    agent.restore_plan(vec![pending_step("old work")]);
    agent.run_turn("continue", &mut ui).await.unwrap();
    assert!(
        agent.plan_incomplete(),
        "continue should keep the pin: {:?}",
        agent.current_plan()
    );

    agent.run_turn("use a BTreeMap", &mut ui).await.unwrap();
    assert!(
        agent.plan_incomplete(),
        "short steering should keep the pin: {:?}",
        agent.current_plan()
    );
}

#[tokio::test]
async fn ask_user_rejects_continue_and_caps_at_one() {
    let mut agent = agent(Vec::new(), config());
    let mut ui = RecUi::default();
    let confirm = agent
        .handle_ask_user(r#"{"question":"should I continue?"}"#, &mut ui)
        .await;
    assert_eq!(confirm.status, hi_tools::ToolStatus::Failed);
    assert!(confirm.content.contains("keep working"));
    assert!(
        ui.ask_user_questions.is_empty(),
        "continue-shaped questions must not open the overlay: {:?}",
        ui.ask_user_questions
    );

    let first = agent
        .handle_ask_user(
            r#"{"question":"REST or gRPC for the public API?"}"#,
            &mut ui,
        )
        .await;
    assert_eq!(first.status, hi_tools::ToolStatus::Succeeded);
    assert_eq!(ui.ask_user_questions.len(), 1);

    let second = agent
        .handle_ask_user(r#"{"question":"And which error type?"}"#, &mut ui)
        .await;
    assert_eq!(second.status, hi_tools::ToolStatus::Failed);
    assert!(second.content.contains("already asked this turn"));
    assert_eq!(ui.ask_user_questions.len(), 1);
}

#[tokio::test]
async fn ask_user_drive_streak_and_what_next_fail_closed() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    agent.begin_drive_turn(crate::DriveKind::Plan);
    let mut ui = RecUi::default();

    let next = agent
        .handle_ask_user(r#"{"question":"what should I do next?"}"#, &mut ui)
        .await;
    assert_eq!(next.status, hi_tools::ToolStatus::Failed);
    assert!(next.content.contains("keep working"));
    assert!(ui.ask_user_questions.is_empty());

    let first = agent
        .handle_ask_user(
            r#"{"question":"REST or gRPC for the public API?"}"#,
            &mut ui,
        )
        .await;
    assert_eq!(first.status, hi_tools::ToolStatus::Succeeded);
    assert_eq!(ui.ask_user_questions.len(), 1);

    agent.ask_user_calls = 0;
    let second = agent
        .handle_ask_user(r#"{"question":"And which error type?"}"#, &mut ui)
        .await;
    assert_eq!(second.status, hi_tools::ToolStatus::Failed);
    assert!(second.content.contains("already asked this drive"));
    assert_eq!(ui.ask_user_questions.len(), 1);
}

#[test]
fn plan_clear_drops_the_pin() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("old work")]);
    let clear = handle_session_command(&mut agent, &Command::Plan("clear".into()), &[])
        .expect("clear effect");
    assert!(agent.current_plan().is_empty());
    assert!(clear.follow_up_prompt.is_none());
    assert!(clear.message.contains("cleared"));

    agent.restore_plan(vec![pending_step("old work")]);
    let replace = handle_session_command(&mut agent, &Command::Plan("replace".into()), &[])
        .expect("replace effect");
    assert!(agent.current_plan().is_empty());
    assert!(replace.message.contains("update_plan"));
}

#[test]
fn plan_pause_and_resume_are_reserved() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    let pause = handle_session_command(&mut agent, &Command::Plan("pause".into()), &[])
        .expect("pause effect");
    assert!(agent.plan_drive_paused());
    assert!(pause.follow_up_prompt.is_none());
    assert!(pause.message.contains("paused"));

    let outcome = completed_outcome(agent.leftover_work());
    assert!(!agent.drive_decision(Some(&outcome)).should_enqueue());

    let status = handle_session_command(&mut agent, &Command::Plan("status".into()), &[])
        .expect("status effect");
    assert!(status.message.contains("plan drive: paused"));

    let resume = handle_session_command(&mut agent, &Command::Plan("resume".into()), &[])
        .expect("resume effect");
    assert!(!agent.plan_drive_paused());
    assert_eq!(
        resume.follow_up_prompt.as_deref(),
        Some(crate::PLAN_DRIVE_PROMPT)
    );
}

fn goal_agent() -> Agent {
    let mut cfg = config();
    cfg.subagents.long_horizon = true;
    agent(Vec::new(), cfg)
}

#[test]
fn drive_decision_goal_wins_over_plan_and_plan_mode_idles() {
    let mut agent = goal_agent();
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    let outcome = completed_outcome(agent.leftover_work());
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Enqueue(crate::DriveKind::Goal)
    );
    agent.set_plan_mode(true);
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::PlanMode
        }
    );
}

#[test]
fn drive_decision_goal_paused_and_parked() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    let outcome = completed_outcome(agent.leftover_work());
    assert!(
        agent
            .try_set_goal_pause_reason(crate::GoalPauseReason::User)
            .unwrap()
    );
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::GoalPaused
        }
    );
    assert!(
        agent
            .try_set_goal_pause_reason(crate::GoalPauseReason::None)
            .unwrap()
    );
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        agent.note_goal_drive_progress(false);
    }
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::GoalParked
        }
    );
    let goal = agent.structured_goal().expect("goal");
    assert!(!goal.is_paused());
    assert_ne!(goal.pause_reason, crate::GoalPauseReason::Stall);
}

#[test]
fn begin_drive_turn_resumes_paused_and_parked_goal() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    assert!(
        agent
            .try_set_goal_pause_reason(crate::GoalPauseReason::User)
            .unwrap()
    );
    agent.begin_drive_turn(crate::DriveKind::Goal);
    assert!(!agent.structured_goal().unwrap().is_paused());
    assert_eq!(agent.goal_drive_stall(), 0);

    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        agent.note_goal_drive_progress(false);
    }
    assert_eq!(agent.goal_drive_status(), "parked");
    agent.begin_drive_turn(crate::DriveKind::Goal);
    assert_eq!(agent.goal_drive_stall(), 0);
    assert_eq!(agent.goal_drive_status(), "running");
}

#[test]
fn interactive_drive_turn_demotes_always_and_restores() {
    let mut agent = goal_agent();
    agent.set_interactive_session(true);
    agent.set_permission_mode(crate::PermissionMode::Always);
    agent.begin_drive_turn(crate::DriveKind::Plan);
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Auto);
    agent.finish_drive_turn();
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Always);

    agent.set_interactive_session(false);
    agent.set_permission_mode(crate::PermissionMode::Always);
    agent.begin_drive_turn(crate::DriveKind::Goal);
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Always);
}

#[test]
fn ingest_checkbox_plan_becomes_sub_goals_without_planner() {
    let workspace = IsolatedWorkspace::new("ingest-checkbox");
    std::fs::write(
        workspace.path("plan.md"),
        "- [x] already shipped\n- [ ] wire the CLI\n- [ ] pass tests\n",
    )
    .unwrap();
    let mut cfg = workspace.config();
    cfg.paths.workspace_root = std::fs::canonicalize(&cfg.paths.workspace_root).unwrap();
    cfg.subagents.long_horizon = true;
    let agent = agent(Vec::new(), cfg);
    let goal = agent
        .try_ingest_goal("implement plan.md")
        .expect("checklist ingest");
    assert_eq!(goal.sub_goals.len(), 3);
    assert_eq!(goal.sub_goals[0].status, crate::GoalStatus::Done);
    assert_eq!(goal.sub_goals[1].status, crate::GoalStatus::Active);
    assert_eq!(goal.sub_goals[1].description, "wire the CLI");
}

#[test]
fn ingest_prose_plan_falls_back_to_planner() {
    let workspace = IsolatedWorkspace::new("ingest-prose");
    std::fs::write(
        workspace.path("plan.md"),
        "# Design\n\nThis document describes the architecture in prose.\n",
    )
    .unwrap();
    let mut cfg = workspace.config();
    cfg.paths.workspace_root = std::fs::canonicalize(&cfg.paths.workspace_root).unwrap();
    cfg.subagents.long_horizon = true;
    let agent = agent(Vec::new(), cfg);
    assert!(agent.try_ingest_goal("implement plan.md").is_none());
}

#[test]
fn goal_drive_stall_skips_stuck_step_and_keeps_driving() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["first".into(), "second".into(), "third".into()],
            )))
            .unwrap()
    );
    let outcome = completed_outcome(agent.leftover_work());
    let mut last = crate::GoalDriveProgress::Unchanged;
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        last = agent.note_goal_drive_progress(false);
    }
    assert!(
        matches!(
            last,
            crate::GoalDriveProgress::Skipped { ref failed, .. } if failed == "first"
        ),
        "{last:?}"
    );
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Enqueue(crate::DriveKind::Goal)
    );
    assert_eq!(agent.goal_drive_stall(), 0);
    let goal = agent.structured_goal().expect("goal");
    assert_eq!(goal.sub_goals[0].status, crate::GoalStatus::Failed);
    assert_eq!(goal.sub_goals[1].status, crate::GoalStatus::Active);
    assert_eq!(goal.pause_reason, crate::GoalPauseReason::None);
}

#[test]
fn goal_drive_two_skips_without_completion_parks() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["first".into(), "second".into(), "third".into()],
            )))
            .unwrap()
    );
    let outcome = completed_outcome(agent.leftover_work());
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        agent.note_goal_drive_progress(false);
    }
    let mut last = crate::GoalDriveProgress::Unchanged;
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        last = agent.note_goal_drive_progress(false);
    }
    assert_eq!(last, crate::GoalDriveProgress::Parked);
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::GoalParked
        }
    );
    let goal = agent.structured_goal().expect("goal");
    assert!(goal.is_thrashing());
    assert_eq!(goal.pause_reason, crate::GoalPauseReason::None);
    assert_eq!(goal.sub_goals[0].status, crate::GoalStatus::Failed);
    assert_eq!(goal.sub_goals[1].status, crate::GoalStatus::Failed);
}
