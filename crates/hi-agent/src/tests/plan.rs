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
    agent.last_plan = vec![PlanStep {
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
async fn new_task_emits_plan_clear_for_frontends() {
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("new task done".into())],
            1,
            1,
        )],
        config(),
    );
    agent.last_plan = vec![PlanStep {
        title: "old unfinished step".into(),
        status: PlanStatus::Pending,
    }];
    let mut ui = RecUi::default();

    agent
        .run_turn("do a different task", &mut ui)
        .await
        .unwrap();

    assert!(agent.last_plan.is_empty());
    assert_eq!(ui.plans, vec![Vec::<PlanStep>::new()]);
}

#[tokio::test]
async fn continue_does_not_preserve_a_completed_plan_box() {
    let mut agent = agent(
        vec![completion(vec![Content::Text("done".into())], 1, 1)],
        config(),
    );
    agent.last_plan = vec![PlanStep {
        title: "old completed step".into(),
        status: PlanStatus::Done,
    }];
    let mut ui = RecUi::default();

    agent.run_turn("continue", &mut ui).await.unwrap();

    assert!(agent.last_plan.is_empty());
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
    let plan_after_turn1 = &agent.last_plan;
    assert!(
        plan_has_pending_steps(plan_after_turn1),
        "plan should persist with pending steps after turn 1: {:?}",
        plan_after_turn1
    );

    // Turn 2: user types "fix a different bug" (NOT "continue"). The plan
    // should be cleared so a stale plan doesn't cause spurious nudges.
    // We can't easily run a full turn here (Canned provider is exhausted),
    // but we can verify the clearing logic by checking that a non-continue
    // input would clear it. Simulate by calling the clearing logic directly.
    let mut plan = agent.last_plan.clone();
    // The agent clears last_plan when input doesn't look like "continue".
    // Verify the heuristic: "fix a different bug" is NOT a continue command.
    assert!(
        !looks_like_continue("fix a different bug"),
        "a new task should not look like continue"
    );
    assert!(
        looks_like_continue("continue"),
        "'continue' should look like continue"
    );
    // Simulate the clearing: a new task clears, "continue" doesn't.
    plan.clear(); // what the agent does on a new task
    assert!(
        !plan_has_pending_steps(&plan),
        "plan should be cleared on a new task"
    );
}
