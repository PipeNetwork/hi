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
async fn runs_a_tool_then_finishes() {
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "1".into(),
                name: "bash".into(),
                arguments: "{\"command\":\"echo hi\"}".into(),
            }],
            5,
            1,
        ),
        completion(vec![Content::Text("all done".into())], 6, 2),
    ];
    let mut agent = agent(responses, config());
    agent.run_turn("do it", &mut NullUi).await.unwrap();

    let roles: Vec<Role> = agent.messages().iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![
            Role::System,
            Role::User,
            Role::Assistant, // tool call
            Role::Tool,      // tool result
            Role::Assistant, // final text
        ]
    );
    // Token totals accumulate across both model calls.
    assert_eq!(agent.totals().input_tokens, 11);
    assert_eq!(agent.totals().output_tokens, 3);
    assert_eq!(agent.messages().last().unwrap().text(), "all done");
}

#[tokio::test]
async fn batched_read_only_tools_run_and_preserve_order() {
    // One round emits two read-only calls; both run (concurrently) and their
    // results are recorded back in call order. Reads resolve against the
    // crate dir (cargo sets cwd to the manifest dir).
    let responses = vec![
        completion(
            vec![
                Content::ToolCall {
                    id: "1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"Cargo.toml"}"#.into(),
                },
                Content::ToolCall {
                    id: "2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"src/lib.rs"}"#.into(),
                },
            ],
            5,
            1,
        ),
        completion(vec![Content::Text("done".into())], 6, 2),
    ];
    let mut agent = agent(responses, config());
    agent.run_turn("scan", &mut NullUi).await.unwrap();

    let outputs: Vec<String> = agent
        .messages()
        .iter()
        .flat_map(|m| &m.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 2, "both tool results recorded");
    assert!(
        outputs[0].contains("hi-agent"),
        "first result is Cargo.toml"
    );
    assert!(
        // The file's top-of-module doc comment — stable in the kept head even
        // after the per-result cap clips this (large) file's middle.
        outputs[1].contains("The agent loop"),
        "second result is lib.rs"
    );
}

#[tokio::test]
async fn compact_replaces_history_with_summary() {
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let responses = vec![completion(
        vec![Content::Text(
            "BRIEF: ported the parser; tests green".into(),
        )],
        7,
        5,
    )];
    let mut agent = agent(responses, config());
    agent.set_session(Box::new(RecordingSession {
        records: records.clone(),
    }));
    // Some history to compact.
    agent.messages_mut().push(Message::user("hello"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("hi".into())]));

    agent.compact(&mut NullUi).await.unwrap();

    // History collapses to system + summary.
    assert_eq!(agent.messages().len(), 2);
    assert_eq!(agent.messages()[0].role, Role::System);
    assert!(
        agent.messages()[1]
            .text()
            .contains("BRIEF: ported the parser"),
        "summary message retained"
    );
    // The summarization call's usage is counted.
    assert_eq!(agent.totals().output_tokens, 5);
    assert_eq!(
        *records.lock().unwrap(),
        vec![(
            Usage {
                input_tokens: 7,
                output_tokens: 5,
                ..Default::default()
            },
            None,
        )],
        "manual compaction persists usage even though compacted messages are transient"
    );
}

#[tokio::test]
async fn hybrid_keeps_recent_and_folds_summary() {
    let mut agent = agent(
        vec![completion(vec![Content::Text("OLD SUMMARY".into())], 3, 2)],
        config(),
    );
    // Two user turns; keep_recent = 1 summarizes the first, keeps the second.
    agent.messages_mut().push(Message::user("q1"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a1".into())]));
    agent.messages_mut().push(Message::user("q2"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a2".into())]));

    agent
        .compact_with(CompactionKind::Hybrid { keep_recent: 1 }, &mut NullUi)
        .await
        .unwrap();

    let m = agent.messages();
    // system + (summary folded into kept user turn) + kept assistant reply.
    assert_eq!(m.len(), 3);
    assert_eq!(m[0].role, Role::System);
    assert_eq!(m[1].role, Role::User);
    assert!(
        m[1].text().contains("OLD SUMMARY"),
        "summary folded: {}",
        m[1].text()
    );
    assert!(
        m[1].text().contains("q2"),
        "recent turn kept: {}",
        m[1].text()
    );
    assert_eq!(m[2].text(), "a2");
    // No two consecutive same-role messages (provider-safe).
    assert!(
        m.windows(2).all(|w| w[0].role != w[1].role),
        "roles must alternate"
    );
}

#[tokio::test]
async fn elide_then_summarize_tail_elides_tool_turns_summarizes_qa() {
    // A session with: an old tool-bearing turn (q1 + read + big result), an
    // old Q&A turn (q2 + text), and a recent turn (q3). The new default
    // strategy should elide the old tool result (keep the call/result
    // skeleton) and summarize only the old Q&A tail, folding the summary
    // into the first kept turn. The recent turn stays verbatim.
    let mut agent = agent(
        vec![completion(vec![Content::Text("QA SUMMARY".into())], 1, 1)],
        config(),
    );
    // Old tool-bearing turn.
    agent.messages_mut().push(Message::user("q1"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("c1", "x".repeat(500)));
    // Old Q&A turn (no tool results) — this is the conversational tail.
    agent.messages_mut().push(Message::user("q2"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a2".into())]));
    // Recent turn.
    agent.messages_mut().push(Message::user("q3"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a3".into())]));

    agent
        .compact_with(
            CompactionKind::ElideThenSummarizeTail { keep_recent: 1 },
            &mut NullUi,
        )
        .await
        .unwrap();

    let m = agent.messages();
    // The old tool result must be elided (skeleton kept, not wiped).
    let tool_results: Vec<&str> = m
        .iter()
        .flat_map(|msg| &msg.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        tool_results.iter().any(|o| o.starts_with("[elided")),
        "old tool result elided (skeleton kept): {tool_results:?}"
    );
    assert!(
        !tool_results.iter().any(|o| o.contains(&"x".repeat(100))),
        "old tool output content gone: {tool_results:?}"
    );
    // The Q&A summary is folded into the first kept turn (q3), and q3 stays.
    let user_texts: Vec<String> = m
        .iter()
        .filter(|msg| msg.role == Role::User)
        .map(|msg| msg.text())
        .collect();
    assert!(
        user_texts.iter().any(|t| t.contains("QA SUMMARY")),
        "Q&A tail summarized and folded: {user_texts:?}"
    );
    assert!(
        user_texts.iter().any(|t| t.contains("q3")),
        "recent turn kept: {user_texts:?}"
    );
    // Provider-safe: roles alternate.
    assert!(
        m.windows(2).all(|w| w[0].role != w[1].role),
        "roles must alternate: {:?}",
        m.iter().map(|x| x.role).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn elide_then_summarize_tail_skips_model_call_when_no_qa_tail() {
    // A pure tool-heavy session (no old Q&A turns): the strategy should
    // elide and NOT make a summarizing model call. Provide no canned
    // completion — if it tried to summarize, the provider would panic on
    // an empty response list.
    let mut agent = agent(vec![], config());
    agent.messages_mut().push(Message::user("q1"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("c1", "x".repeat(500)));
    agent.messages_mut().push(Message::user("q2"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a2".into())]));

    // keep_recent = 1 → q2 is recent; q1's tool result is old and gets
    // elided. No Q&A tail older than q2 → no model call.
    agent
        .compact_with(
            CompactionKind::ElideThenSummarizeTail { keep_recent: 1 },
            &mut NullUi,
        )
        .await
        .unwrap();
    let m = agent.messages();
    let tool_results: Vec<&str> = m
        .iter()
        .flat_map(|msg| &msg.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        tool_results.iter().any(|o| o.starts_with("[elided")),
        "old tool result elided: {tool_results:?}"
    );
}

#[tokio::test]
async fn record_decision_persists_across_compaction_in_system_prompt() {
    // A decision recorded via the tool survives a compaction in the system
    // prompt (the log is injected into the system message, which compaction
    // preserves verbatim — not summarized away).
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "d1".into(),
                name: "record_decision".into(),
                arguments: r#"{"summary":"use BTreeMap","rationale":"ordered iteration","files":["src/m.rs"]}"#.into(),
            }],
            1,
            1,
        ),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    agent.run_turn("refactor", &mut NullUi).await.unwrap();
    assert_eq!(agent.decisions().entries().len(), 1);
    assert_eq!(agent.decisions().entries()[0].summary, "use BTreeMap");

    // The system prompt contains the decision.
    let sys = agent.messages()[0].text();
    assert!(
        sys.contains("use BTreeMap") && sys.contains("ordered iteration"),
        "decision in system prompt: {sys}"
    );

    // A compaction that summarizes the Q&A tail must NOT remove the
    // decision from the system prompt.
    agent
        .compact_with(CompactionKind::Summarize, &mut NullUi)
        .await
        .unwrap();
    let sys_after = agent.messages()[0].text();
    assert!(
        sys_after.contains("use BTreeMap"),
        "decision survives compaction: {sys_after}"
    );
}

#[tokio::test]
async fn proactive_verify_surfaces_a_per_edit_check_failure() {
    // With proactive_verify on, a write to a .py file with a syntax error
    // triggers a background `python3 -m py_compile` whose failure surfaces
    // as a status line during the turn (before turn-end verify). Skipped if
    // python3 isn't on PATH (the check just won't run).
    if std::process::Command::new("sh")
        .arg("-c")
        .arg("command -v python3")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping: python3 not on PATH");
        return;
    }
    let _guard = VERIFY_TEST_LOCK.lock().await;
    let mut cfg = config();
    cfg.proactive_verify = true;
    let tmp = temp_file("proactive");
    let py = tmp.with_extension("py");
    let p = py.to_string_lossy().to_string();
    // Write invalid Python so py_compile fails.
    let responses = vec![
        Completion {
            content: vec![Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: format!(r#"{{"path":{p:?},"content":"def (\n"}}"#),
            }],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                context_occupancy: 1,
                ..Default::default()
            },
            stop_reason: None,
        },
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("write it", &mut ui).await.unwrap();
    let _ = std::fs::remove_file(&py);
    // A proactive-check failure status line names the file.
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("proactive check failed") && s.contains(&p)),
        "proactive failure surfaced: {:?}",
        ui.statuses
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

#[tokio::test]
async fn scheduler_parallelism_counts_concurrent_batches() {
    // A batch of independent reads (different paths, no deps) should run
    // concurrently — telemetry reports max_concurrent_batch > 1 and a
    // sub-100% serial share. Pins that the dep-aware scheduler's
    // concurrency is measured, not just shipped on faith.
    let cfg = config();
    let responses = vec![
        completion(
            vec![
                Content::ToolCall {
                    id: "r1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"a.rs"}"#.into(),
                },
                Content::ToolCall {
                    id: "r2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"b.rs"}"#.into(),
                },
                Content::ToolCall {
                    id: "r3".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"c.rs"}"#.into(),
                },
            ],
            1,
            1,
        ),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("read them", &mut ui).await.unwrap();
    let tel = agent.last_turn_telemetry();
    assert_eq!(tel.tool_calls, 3, "three reads ran: {:?}", tel);
    assert!(
        tel.max_concurrent_batch >= 2,
        "independent reads overlapped: {:?}",
        tel
    );
    assert!(
        tel.serial_runs < tel.tool_calls,
        "not all serial: {:?}",
        tel
    );
    // The timeline records each call with its tool name and path.
    assert_eq!(
        tel.tool_timeline.len(),
        3,
        "timeline has one entry per call: {:?}",
        tel.tool_timeline
    );
    let tools: Vec<&str> = tel.tool_timeline.iter().map(|e| e.tool.as_str()).collect();
    assert!(tools.iter().all(|&t| t == "read"), "all reads: {tools:?}");
    let paths: Vec<&str> = tel.tool_timeline.iter().map(|e| e.path.as_str()).collect();
    assert!(
        paths.contains(&"a.rs") && paths.contains(&"b.rs") && paths.contains(&"c.rs"),
        "timeline paths match calls: {paths:?}"
    );
    assert!(
        tel.tool_timeline.iter().all(|e| e.error),
        "reads error (files don't exist in test): {:?}",
        tel.tool_timeline
    );
}

#[tokio::test]
async fn hybrid_falls_back_to_summarize_when_too_few_turns() {
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("WHOLE SUMMARY".into())],
            1,
            1,
        )],
        config(),
    );
    agent.messages_mut().push(Message::user("only turn"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a".into())]));
    // keep_recent = 3 but only one turn → no recent window → summarize all.
    agent
        .compact_with(CompactionKind::Hybrid { keep_recent: 3 }, &mut NullUi)
        .await
        .unwrap();
    let m = agent.messages();
    assert_eq!(m.len(), 2);
    assert_eq!(m[0].role, Role::System);
    assert!(m[1].text().contains("WHOLE SUMMARY"));
}

#[tokio::test]
async fn elide_shrinks_old_tool_output_without_a_model_call() {
    // Empty provider: if elision tried to call the model, this would panic.
    let mut agent = agent(vec![], config());
    let big = "x".repeat(500);
    agent.messages_mut().push(Message::user("read a"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("c1", big.clone()));
    agent.messages_mut().push(Message::user("read b")); // recent turn
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "c2".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("c2", big.clone()));

    agent
        .compact_with(
            CompactionKind::ElideToolOutput { keep_recent: 1 },
            &mut NullUi,
        )
        .await
        .unwrap();

    let outputs: Vec<String> = agent
        .messages()
        .iter()
        .flat_map(|m| &m.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect();
    assert!(
        outputs[0].starts_with("[elided"),
        "old elided: {}",
        outputs[0]
    );
    assert_eq!(outputs[1], big, "recent kept verbatim");
}

