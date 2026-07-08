use super::common::*;
use super::*;

struct FailingCheckpointSession;

impl SessionSink for FailingCheckpointSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_checkpoints(&mut self, _refs: &[String]) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("disk full"))
    }
}

#[test]
fn resume_restores_retained_checkpoint_refs() {
    let checkpoints = (0..55).map(|i| format!("{i:040x}")).collect::<Vec<_>>();

    let agent = Agent::resume(
        Box::new(Canned(Mutex::new(Vec::new()))),
        config(),
        vec![Message::system("system")],
        Usage::default(),
        checkpoints,
        None,
        DecisionLog::default(),
    );

    assert_eq!(
        agent.checkpoint_count(),
        MAX_CHECKPOINTS,
        "resume keeps the retained checkpoint stack, capped to the undo limit"
    );
}

#[tokio::test]
async fn undo_keeps_checkpoint_when_restore_fails() {
    let mut agent = agent(vec![], config());
    agent.checkpoints.push("not-a-valid-checkpoint".to_string());

    let err = agent.undo().await.unwrap_err();

    assert!(!err.to_string().is_empty(), "expected restore error");
    assert_eq!(
        agent.checkpoint_count(),
        1,
        "failed restore should leave the checkpoint available for retry"
    );
}

#[tokio::test]
async fn undo_keeps_checkpoint_when_persisting_shortened_stack_fails() {
    let untracked = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()
        .map(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .any(|line| line.starts_with("?? "))
        })
        .unwrap_or(false);
    if untracked {
        return;
    }

    let Some(checkpoint) = hi_tools::checkpoint::create(std::path::Path::new(".")).await else {
        return;
    };
    let mut agent = agent(vec![], config());
    agent.checkpoints.push(checkpoint);
    agent.set_session(Box::new(FailingCheckpointSession));

    let err = agent.undo().await.unwrap_err();

    if !err.to_string().contains("disk full") {
        assert_eq!(
            agent.checkpoint_count(),
            1,
            "checkpoint stack should stay live when restore fails before checkpoint persistence"
        );
        return;
    }
    assert_eq!(
        agent.checkpoint_count(),
        1,
        "checkpoint stack should stay live when the shortened stack cannot be persisted"
    );
}

#[tokio::test]
async fn tools_unavailable_fast_path_resets_state_and_shows_message() {
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let mut cfg = config();
    cfg.tool_mode = ToolMode::ChatOnly;
    let mut agent = agent(vec![], cfg);
    agent.last_verify = Some(true);
    agent.last_changed_files = vec!["old.rs".to_string()];
    agent.last_compat_fallbacks = vec!["compat fallback".to_string()];
    agent.last_turn_telemetry = TurnTelemetry {
        repeat_nudges: 7,
        stalled_unfinished: true,
        tool_calls: 3,
        ..TurnTelemetry::default()
    };
    agent.last_plan = vec![PlanStep {
        title: "stale step".to_string(),
        status: PlanStatus::Active,
    }];
    agent
        .messages_mut()
        .push(Message::user("[hi:nudge:repeat] stale nudge 1"));
    agent
        .messages_mut()
        .push(Message::user("[hi:nudge:continue] stale nudge 2"));
    agent
        .messages_mut()
        .push(Message::user("[hi:nudge:verify] stale nudge 3"));
    agent.persisted = agent.messages().len();
    agent.set_session(Box::new(RecordingSession {
        records: records.clone(),
    }));
    let mut ui = RecUi::default();

    agent
        .run_turn("fix the crash in src/main.rs", &mut ui)
        .await
        .unwrap();

    assert_eq!(agent.last_verify(), None);
    assert!(agent.last_changed_files().is_empty());
    assert!(agent.last_compat_fallbacks().is_empty());
    assert_eq!(agent.last_turn_telemetry(), &TurnTelemetry::default());
    assert!(agent.last_plan.is_empty());
    agent.messages.validate_for_provider().unwrap();
    assert!(
        !agent
            .messages()
            .iter()
            .any(|message| message.text().contains("[hi:nudge:")),
        "stale synthetic nudges should be stripped before recording the blocked turn: {:?}",
        agent
            .messages()
            .iter()
            .map(|message| (message.role, message.text()))
            .collect::<Vec<_>>()
    );
    assert_eq!(agent.persisted, agent.messages().len());
    assert_eq!(
        records.lock().unwrap().len(),
        1,
        "blocked turn should persist without a stale persisted index"
    );
    assert!(
        ui.assistant.trim().is_empty(),
        "tools-disabled guardrail should not emit assistant text, got: {:?}",
        ui.assistant
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("tool mode") && status.contains("blocks")),
        "tools-disabled error should be visible, got: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn resume_repairs_provider_invisible_assistant_before_request() {
    let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = ScriptedProvider {
        steps: Mutex::new(vec![ProviderStep::Completion(completion(
            vec![Content::Text("ok".into())],
            1,
            1,
        ))]),
        requests: requests.clone(),
        max_tokens: None,
    };
    let history = vec![
        Message::system("system"),
        Message::user("old prompt"),
        Message::assistant(vec![
            Content::Text(String::new()),
            Content::Thinking {
                text: "unsigned thinking".into(),
                signature: None,
            },
        ]),
    ];
    let mut agent = Agent::resume(
        Box::new(provider),
        config(),
        history,
        Usage::default(),
        Vec::new(),
        None,
        DecisionLog::default(),
    );
    let mut ui = RecUi::default();

    agent.run_turn("next question", &mut ui).await.unwrap();

    agent.messages.validate_for_provider().unwrap();
    let requests = requests.lock().unwrap();
    let sent = requests.first().expect("provider request recorded");
    let repaired = sent
        .iter()
        .find(|message| message.role == Role::Assistant)
        .expect("resumed assistant message sent");
    assert!(
        repaired
            .content
            .iter()
            .any(|c| matches!(c, Content::Text(t) if !t.trim().is_empty())),
        "resumed provider-invisible assistant message should be repaired before request: {repaired:?}"
    );
}

#[tokio::test]
async fn resume_repairs_out_of_order_tool_results_before_request() {
    let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = ScriptedProvider {
        steps: Mutex::new(vec![ProviderStep::Completion(completion(
            vec![Content::Text("ok".into())],
            1,
            1,
        ))]),
        requests: requests.clone(),
        max_tokens: None,
    };
    let history = vec![
        Message::system("system"),
        Message::user("old prompt"),
        Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "bash".into(),
            arguments: "{}".into(),
        }]),
        Message::assistant(vec![Content::Text("interposed answer".into())]),
        Message::tool_result("c1", "late result"),
    ];
    let mut agent = Agent::resume(
        Box::new(provider),
        config(),
        history,
        Usage::default(),
        Vec::new(),
        None,
        DecisionLog::default(),
    );
    let mut ui = RecUi::default();

    agent.run_turn("next question", &mut ui).await.unwrap();

    agent.messages.validate_for_provider().unwrap();
    let requests = requests.lock().unwrap();
    let sent = requests.first().expect("provider request recorded");
    assert!(
        sent.iter().all(|message| message.role != Role::Tool
            && message
                .content
                .iter()
                .all(|content| !matches!(content, Content::ToolCall { .. }))),
        "out-of-order legacy tool skeleton should be repaired before request: {sent:?}"
    );
    assert!(
        sent.windows(2)
            .all(|pair| !(pair[0].role == Role::Assistant && pair[1].role == Role::Assistant)),
        "stripping an unsafe tool skeleton should not leave adjacent assistant turns: {sent:?}"
    );
}

#[tokio::test]
async fn resume_repairs_consecutive_user_messages_before_request() {
    let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = ScriptedProvider {
        steps: Mutex::new(vec![ProviderStep::Completion(completion(
            vec![Content::Text("ok".into())],
            1,
            1,
        ))]),
        requests: requests.clone(),
        max_tokens: None,
    };
    let history = vec![
        Message::system("system"),
        Message::user("legacy user one"),
        Message::user("legacy user two"),
        Message::assistant(vec![Content::Text("old answer".into())]),
    ];
    let mut agent = Agent::resume(
        Box::new(provider),
        config(),
        history,
        Usage::default(),
        Vec::new(),
        None,
        DecisionLog::default(),
    );
    let mut ui = RecUi::default();

    agent.run_turn("next question", &mut ui).await.unwrap();

    agent.messages.validate_for_provider().unwrap();
    let requests = requests.lock().unwrap();
    let sent = requests.first().expect("provider request recorded");
    assert!(
        sent.windows(2)
            .all(|pair| !(pair[0].role == Role::User && pair[1].role == Role::User)),
        "resumed request should not contain adjacent user messages: {sent:?}"
    );
    assert!(
        sent.iter().any(|message| message.role == Role::User
            && message.text().contains("legacy user one")
            && message.text().contains("legacy user two")),
        "legacy adjacent users should be folded together before send: {sent:?}"
    );
}

#[tokio::test]
async fn nudges_when_model_repeats_the_same_command() {
    // The model runs a command, then re-issues the *exact same* call next
    // round. The repetition guard nudges it to act on the output instead of
    // re-running, and the model then finishes. One repeat-nudge, no
    // "stuck repeating" notice.
    let responses = vec![
        echo_call(),
        echo_call(), // exact repeat → nudged
        completion(vec![Content::Text("Done. Run cargo test.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("check it", &mut ui).await.unwrap();
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("re-ran the same command"))
            .count(),
        1,
        "exactly one repeat-nudge, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("kept re-running")),
        "no stuck-repeating notice once it moved on, got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn completed");
}

#[tokio::test]
async fn gives_up_with_notice_after_repeat_cap() {
    // The model re-issues the exact same command every round, through the
    // whole repeat-nudge budget: bounded nudges, then one chat-only final
    // answer recovery attempt. If the model still emits tools, the turn stops
    // incomplete instead of running to the step cap.
    let mut responses = vec![echo_call()];
    for _ in 0..(config().max_repeat_nudges + 1) {
        responses.push(echo_call()); // exact repeat each round
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("check it", &mut ui).await.unwrap();
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("re-ran the same command"))
            .count(),
        config().max_repeat_nudges as usize,
        "repeat-nudges are bounded, got: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("turn stopped incomplete")),
        "incomplete notice after forced final recovery, got: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 1);
    agent.messages.validate_for_provider().unwrap();
    assert!(
        agent
            .messages()
            .iter()
            .filter(|m| m.role == hi_ai::Role::Assistant)
            .all(|m| !m.content.is_empty()),
        "skipped repeated tool-call turns must not leave empty assistant messages: {:?}",
        agent
            .messages()
            .iter()
            .map(|m| (m.role, m.content.len(), m.text()))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn gives_up_when_bash_only_cycles_through_stop_words() {
    let mut cfg = config();
    cfg.max_repeat_nudges = 1;
    let responses = vec![
        bash_completion("echo stop"),
        bash_completion("echo quit"),
        bash_completion("echo exit"),
        bash_completion("echo done"),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    agent.run_turn("stop when complete", &mut ui).await.unwrap();

    assert_eq!(
        ui.tool_results.len(),
        2,
        "first semantic repeat gets grace, later no-op bash calls are skipped: {:?}",
        ui.tool_results
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("no-op shell commands")),
        "expected no-op bash loop nudge/status, got: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("turn stopped incomplete")),
        "guard should stop the turn without waiting for max_steps: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("reached step limit")),
        "semantic no-progress guard should fire before step cap: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().repeat_nudges, 1);
    assert!(agent.last_turn_telemetry().stalled_unfinished);
    assert!(ui.assistant.trim().is_empty());
    agent.messages.validate_for_provider().unwrap();
}

#[tokio::test]
async fn useful_distinct_bash_commands_are_not_no_progress_bounded() {
    let responses = vec![
        bash_completion("pwd"),
        bash_completion("echo hi"),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("run two harmless shell checks", &mut ui)
        .await
        .unwrap();

    assert_eq!(ui.tool_results.len(), 2, "normal bash calls still run");
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("no-op shell commands")),
        "normal bash calls should not hit the no-op guard: {:?}",
        ui.statuses
    );
    assert_eq!(agent.messages().last().unwrap().text(), "done");
}

#[tokio::test]
async fn repeated_no_progress_nudges_force_one_chat_only_final_answer() {
    let mut cfg = config();
    cfg.max_repeat_nudges = 2;
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(vec![
            bash_completion("echo stop"),
            bash_completion("echo quit"),
            bash_completion("echo exit"),
            bash_completion("echo done"),
            completion(
                vec![Content::Text(
                    "Stopped after the available no-op output.".into(),
                )],
                1,
                1,
            ),
        ]),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(Box::new(provider), cfg);
    let mut ui = RecUi::default();

    agent.run_turn("stop when complete", &mut ui).await.unwrap();

    assert!(
        ui.assistant.contains("Stopped after"),
        "forced final answer should be surfaced, got: {}",
        ui.assistant
    );
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 1);
    assert!(!agent.last_turn_telemetry().hit_step_cap);
    assert_eq!(
        modes.lock().unwrap().last(),
        Some(&ToolMode::ChatOnly),
        "the recovery attempt should be chat-only"
    );
}

#[derive(Default)]
struct DenyEditsUi {
    confirm_calls: usize,
    tool_results: Vec<(String, String)>,
    turn_end: Option<String>,
}

impl Ui for DenyEditsUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn confirm_edit(&mut self, _: &str, _: &str) -> bool {
        self.confirm_calls += 1;
        false
    }
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, name: &str, result: &str) {
        self.tool_results
            .push((name.to_string(), result.to_string()));
    }
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, summary: &str) {
        self.turn_end = Some(summary.to_string());
    }
}

#[tokio::test]
async fn denied_edit_counts_as_completed_for_dependent_calls() {
    let path = temp_file("denied-edit-dependent-read");
    let p = path.to_string_lossy().to_string();
    let response = completion(
        vec![
            Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: serde_json::json!({ "path": p.clone(), "content": "new" }).to_string(),
            },
            Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": p }).to_string(),
            },
        ],
        1,
        1,
    );
    let mut cfg = config();
    cfg.confirm_edits = true;
    let mut agent = agent(
        vec![
            response,
            completion(vec![Content::Text("Done.".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = DenyEditsUi::default();

    agent.run_turn("check it", &mut ui).await.unwrap();

    assert_eq!(ui.confirm_calls, 1);
    assert_eq!(
        ui.tool_results
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["write", "read"]
    );
    assert!(ui.tool_results[0].1.contains("Edit skipped by user"));
    assert!(
        !agent
            .messages()
            .iter()
            .any(|message| message.text().contains("[tool result missing]")),
        "denied calls should be accounted for without synthesized missing results"
    );
    agent.messages.validate_for_provider().unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn repeated_successful_background_output_poll_is_not_repeat_nudged() {
    let started = hi_tools::execute(
        "bash",
        r#"{"command":"printf bg-live; sleep 1","run_in_background":true}"#,
    )
    .await;
    let id = started
        .content
        .split('`')
        .nth(1)
        .expect("handle id in start message")
        .to_string();
    assert!(id.starts_with("bg_"), "got: {}", started.content);
    let bash_output = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: "bo".into(),
                name: "bash_output".into(),
                arguments: serde_json::json!({ "id": id }).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        bash_output(&id),
        bash_output(&id),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("watch the background job", &mut ui)
        .await
        .unwrap();

    let _ = hi_tools::execute("bash_kill", &format!(r#"{{"id":"{id}"}}"#)).await;
    let bash_output_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "bash_output")
        .count();
    assert_eq!(
        bash_output_results, 2,
        "successful background polls are time-dependent and should both execute: {:?}",
        ui.tool_results
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("re-ran the same command")
                || s.contains("kept polling stale background process handles")),
        "successful background polls should not be repeat-nudged: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn repeated_completed_background_output_poll_is_bounded() {
    let started = hi_tools::execute(
        "bash",
        r#"{"command":"printf bg-complete","run_in_background":true}"#,
    )
    .await;
    let id = started
        .content
        .split('`')
        .nth(1)
        .expect("handle id in start message")
        .to_string();
    assert!(id.starts_with("bg_"), "got: {}", started.content);
    let args = format!(r#"{{"id":"{id}"}}"#);
    let mut terminal_seen = false;
    for _ in 0..50 {
        let out = hi_tools::execute("bash_output", &args).await;
        if out.content.contains(": exited") {
            terminal_seen = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(terminal_seen, "background process should have exited");

    let bash_output = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: "bo".into(),
                name: "bash_output".into(),
                arguments: serde_json::json!({ "id": id }).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        bash_output(&id),
        bash_output(&id),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("check the completed background job", &mut ui)
        .await
        .unwrap();

    let bash_output_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "bash_output")
        .count();
    assert_eq!(
        bash_output_results, 1,
        "completed background handle should be recognized as stale after one poll: {:?}",
        ui.tool_results
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("kept polling stale background process handles")),
        "completed background handle should be repeat-nudged: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn nudges_when_model_cycles_missing_background_outputs() {
    let bash_output = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: "bo".into(),
                name: "bash_output".into(),
                arguments: serde_json::json!({ "id": id }).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        bash_output("bg_missing_1"),
        bash_output("bg_missing_2"),
        bash_output("bg_missing_1"),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("check the background jobs", &mut ui)
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("kept polling stale background process handles")),
        "expected stale background-output nudge, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("re-read files it already inspected")),
        "background-output cycles should not be reported as file re-reads: {:?}",
        ui.statuses
    );
    let bash_output_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "bash_output")
        .count();
    assert_eq!(
        bash_output_results, 2,
        "the repeated missing handle should be skipped, got results: {:?}",
        ui.tool_results
    );
    assert!(ui.turn_end.is_some(), "turn completed after the nudge");
}

#[tokio::test]
async fn nudges_when_model_cycles_missing_background_kills() {
    let bash_kill = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: "bk".into(),
                name: "bash_kill".into(),
                arguments: serde_json::json!({ "id": id }).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        bash_kill("bg_missing_1"),
        bash_kill("bg_missing_2"),
        bash_kill("bg_missing_1"),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("stop the background jobs", &mut ui)
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("kept using stale background process handles")),
        "expected stale background-kill nudge, got: {:?}",
        ui.statuses
    );
    let bash_kill_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "bash_kill")
        .count();
    assert_eq!(
        bash_kill_results, 2,
        "the repeated missing kill handle should be skipped, got results: {:?}",
        ui.tool_results
    );
    assert!(ui.turn_end.is_some(), "turn completed after the nudge");
}

#[tokio::test]
async fn missing_background_output_after_prior_mutation_stalls_instead_of_looping() {
    let path = temp_file("missing-bg-after-mutation");
    let p = path.to_string_lossy().to_string();
    let bash_output = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: "bo".into(),
                name: "bash_output".into(),
                arguments: serde_json::json!({ "id": id }).to_string(),
            }],
            1,
            1,
        )
    };
    let mut responses = vec![
        write_completion(&p),
        bash_output("bg_missing_1"),
        bash_output("bg_missing_2"),
    ];
    for i in 0..(config().max_repeat_nudges + 1) {
        responses.push(bash_output(if i % 2 == 0 {
            "bg_missing_1"
        } else {
            "bg_missing_2"
        }));
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent.run_turn("fix the harness", &mut ui).await.unwrap();

    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("kept polling stale background process handles"))
            .count(),
        config().max_repeat_nudges as usize,
        "repeat nudges should be bounded, got: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("turn stopped incomplete")),
        "expected forced final recovery to stop incomplete, got: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 1);
    let bash_output_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "bash_output")
        .count();
    assert_eq!(
        bash_output_results, 2,
        "stale background polls should not execute after the two failed handles are known: {:?}",
        ui.tool_results
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn missing_background_kill_after_prior_mutation_stalls_instead_of_looping() {
    let path = temp_file("missing-bg-kill-after-mutation");
    let p = path.to_string_lossy().to_string();
    let bash_kill = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: "bk".into(),
                name: "bash_kill".into(),
                arguments: serde_json::json!({ "id": id }).to_string(),
            }],
            1,
            1,
        )
    };
    let mut responses = vec![
        write_completion(&p),
        bash_kill("bg_missing_1"),
        bash_kill("bg_missing_2"),
    ];
    for i in 0..(config().max_repeat_nudges + 1) {
        responses.push(bash_kill(if i % 2 == 0 {
            "bg_missing_1"
        } else {
            "bg_missing_2"
        }));
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent.run_turn("fix the harness", &mut ui).await.unwrap();

    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("kept using stale background process handles"))
            .count(),
        config().max_repeat_nudges as usize,
        "repeat nudges should be bounded, got: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("turn stopped incomplete")),
        "expected forced final recovery to stop incomplete, got: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 1);
    let bash_kill_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "bash_kill")
        .count();
    assert_eq!(
        bash_kill_results, 2,
        "stale background kills should not execute after the two failed handles are known: {:?}",
        ui.tool_results
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn implementation_re_read_exhaustion_reports_incomplete_not_stuck_repeating() {
    // An implementation task where the model reads a file, then keeps
    // re-reading it through the repeat budget and then ignores the
    // implementation repair nudges — the "explore forever, never edit" failure
    // mode. The turn should end with the implementation-incomplete message (so
    // the user knows no edit was made), NOT the generic "stuck repeating"
    // notice or a forced chat-only final answer.
    let path = temp_file("impl-reread-exhaust");
    std::fs::write(&path, "fn parse() {}\n").unwrap();
    let p = path.to_string_lossy().to_string();
    let read = || {
        completion(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": p.clone() }).to_string(),
            }],
            1,
            1,
        )
    };
    // Read once (new evidence), then re-read past the repeat and no-change
    // repair budgets. The guard nudges up to max_repeat_nudges times, spends
    // the implementation no-change repair budget, then stalls on the next
    // non-mutating repeat.
    let mut responses = vec![read()];
    for _ in 0..(config().max_repeat_nudges + 3) {
        responses.push(read());
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent
        .run_turn("/build parser implementation", &mut ui)
        .await
        .unwrap();
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("turn stopped incomplete")),
        "expected implementation repair exhaustion to stop incomplete, got: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 0);
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("repeating without editing")),
        "expected implementation-specific repair nudge, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("kept re-running")),
        "should not use the generic stuck-repeating notice for an impl task, got: {:?}",
        ui.statuses
    );
    assert!(
        ui.assistant.trim().is_empty(),
        "guardrail should not emit canned assistant text, got: {}",
        ui.assistant
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn re_read_after_prior_mutation_does_not_hard_stall_the_turn() {
    // This mirrors long harness work: earlier plan steps already changed files,
    // then a later step gets stuck re-reading inspected context. The no-new-
    // evidence guard should nudge, but after its advisory budget it must allow
    // execution so the harness can continue instead of ending the whole turn as
    // stalled.
    let path = temp_file("reread-after-mutation");
    let p = path.to_string_lossy().to_string();
    let read = || {
        completion(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": p.clone() }).to_string(),
            }],
            1,
            1,
        )
    };
    let mut responses = vec![
        write_completion(&p),
        read(), // first read after the write executes and records evidence
    ];
    for _ in 0..(config().max_repeat_nudges + 1) {
        responses.push(read());
    }
    responses.push(completion(vec![Content::Text("Done.".into())], 1, 1));

    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent
        .run_turn("continue the test extraction", &mut ui)
        .await
        .unwrap();

    assert!(
        ui.turn_end.is_some(),
        "turn should continue after advisory re-read nudges, got statuses: {:?}",
        ui.statuses
    );
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("re-read files it already inspected")
                || s.contains("re-ran the same command"))
            .count(),
        config().max_repeat_nudges as usize,
        "repeat nudges should still be bounded, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("kept re-running"))
            && ui.assistant.trim().is_empty(),
        "prior mutations should not be converted into a hard repeat stall, got statuses {:?} assistant {}",
        ui.statuses,
        ui.assistant
    );
    let read_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "read")
        .count();
    assert!(
        read_results >= 2,
        "a re-read should execute after the advisory budget is spent, got tool results: {:?}",
        ui.tool_results
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn implementation_re_read_cycle_recovers_when_model_edits() {
    // The concrete nudge (naming inspected files + plan step) gives the model
    // a specific action to take. The model re-reads, gets nudged to edit, and
    // then actually makes an edit — the turn should complete successfully, not
    // stall. This verifies the guard pushes the model toward editing without
    // killing the turn prematurely.
    let path = temp_file("impl-reread-recover");
    std::fs::write(&path, "fn parse() {}\n").unwrap();
    let p = path.to_string_lossy().to_string();
    let read = || {
        completion(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": p.clone() }).to_string(),
            }],
            1,
            1,
        )
    };
    let edit = || {
        completion(
            vec![Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: serde_json::json!({
                    "path": p.clone(),
                    "content": "fn parse() -> i32 { 42 }\n"
                })
                .to_string(),
            }],
            1,
            1,
        )
    };
    // Read once (new), re-read once (nudged to edit), then actually edit.
    // The model gets one nudge, then breaks out of the cycle by editing.
    let mut responses = vec![
        read(),
        read(), // re-read → nudge 1/2
        edit(), // model heeds the nudge and edits
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    // Extra fallbacks in case preflight consumes an extra round.
    for _ in 0..4 {
        responses.push(completion(vec![Content::Text("Done.".into())], 1, 1));
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent
        .run_turn("/build parser implementation", &mut ui)
        .await
        .unwrap();
    // The turn completed (the model edited and finished), not stalled.
    assert!(
        ui.turn_end.is_some(),
        "turn should complete after the model edits, got statuses: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("kept re-reading without editing")),
        "should not stall since the model eventually edited, got: {:?}",
        ui.statuses
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn implementation_re_read_nudge_names_inspected_files_and_plan_step() {
    // The implementation re-read nudge must be concrete: it should name the
    // inspected file paths and the next plan step (if any), not just say
    // "start editing" generically. A strong model responds to one concrete
    // nudge; a generic nudge is ignored.
    let path = temp_file("impl-nudge-concrete");
    std::fs::write(&path, "fn parse() {}\n").unwrap();
    let p = path.to_string_lossy().to_string();
    let read = || {
        completion(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": p.clone() }).to_string(),
            }],
            1,
            1,
        )
    };
    let plan = || {
        completion(
            vec![Content::ToolCall {
                id: "p".into(),
                name: "update_plan".into(),
                arguments: serde_json::json!({
                    "steps": [
                        {"title": "Inspect the parser", "status": "done"},
                        {"title": "Fix the parser bug", "status": "pending"},
                    ]
                })
                .to_string(),
            }],
            1,
            1,
        )
    };
    let mut responses = vec![
        plan(), // model makes a plan
        read(), // model reads the file (new evidence)
        read(), // re-read → nudge (should name the file + plan step)
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    // Extra fallbacks for preflight/plan rounds.
    for _ in 0..6 {
        responses.push(completion(vec![Content::Text("Done.".into())], 1, 1));
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent
        .run_turn("/build parser implementation", &mut ui)
        .await
        .unwrap();
    // The nudge is a user message in the transcript — find it and verify it
    // contains the inspected path and the plan step title.
    let nudge_text = agent
        .messages()
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| m.text())
        .find(|t| t.contains("do not re-read") || t.contains("do not re-read them"));
    assert!(
        nudge_text.is_some(),
        "expected a re-read nudge in the transcript, got messages: {:?}",
        agent
            .messages()
            .iter()
            .map(|m| (m.role, m.text().chars().take(80).collect::<String>()))
            .collect::<Vec<_>>()
    );
    let nudge = nudge_text.unwrap();
    assert!(
        nudge.contains(&p),
        "nudge should name the inspected file path, got: {nudge}"
    );
    assert!(
        nudge.contains("Fix the parser bug"),
        "nudge should name the next plan step, got: {nudge}"
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn implementation_repeat_exhaustion_repairs_to_edit_instead_of_forced_final() {
    let inspect_path = temp_file("impl-repeat-inspect");
    std::fs::write(&inspect_path, "fn add(a: i32, b: i32) -> i32 { a - b }\n").unwrap();
    let inspect_path_string = inspect_path.to_string_lossy().to_string();
    let write_path = temp_file("impl-repeat-write");
    let write_path_string = write_path.to_string_lossy().to_string();
    let plan = || {
        completion(
            vec![Content::ToolCall {
                id: "p".into(),
                name: "update_plan".into(),
                arguments: serde_json::json!({
                    "steps": [
                        {"title": "Inspect the bug", "status": "done"},
                        {"title": "Fix the arithmetic", "status": "pending"},
                    ]
                })
                .to_string(),
            }],
            1,
            1,
        )
    };
    let read = || {
        completion(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspect_path_string }).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        plan(),
        read(),
        read(),
        read(),
        read(),
        write_completion(&write_path_string),
        completion(vec![Content::Text("Implemented it.".into())], 1, 1),
        bash_completion("cargo test --help"),
        completion(
            vec![Content::Text(format!(
                "Changed {write_path_string} and validated with cargo test --help."
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("/build parser implementation", &mut ui)
        .await
        .unwrap();

    assert_eq!(std::fs::read_to_string(&write_path).unwrap(), "x");
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 0);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("repeating without editing")),
        "expected implementation repeat repair status: {:?}",
        ui.statuses
    );
    assert!(
        !agent.last_turn_telemetry().stalled_unfinished,
        "turn should recover by editing and validating, statuses: {:?}",
        ui.statuses
    );
    let _ = std::fs::remove_file(inspect_path);
    let _ = std::fs::remove_file(write_path);
}

#[tokio::test]
async fn does_not_nudge_a_different_command() {
    // Two consecutive tool calls with different arguments are not a repeat —
    // both execute, no repeat-nudge.
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "t".into(),
                name: "bash".into(),
                arguments: "{\"command\":\"echo one\"}".into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "t".into(),
                name: "bash".into(),
                arguments: "{\"command\":\"echo two\"}".into(),
            }],
            1,
            1,
        ),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("run them", &mut ui).await.unwrap();
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("re-ran the same command")),
        "different commands are not a repeat, got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn completed");
}

#[tokio::test]
async fn nudges_when_different_inspections_return_the_same_output() {
    let dir_a = temp_file("same-output-dir-a");
    let dir_b = temp_file("same-output-dir-b");
    std::fs::create_dir(&dir_a).unwrap();
    std::fs::create_dir(&dir_b).unwrap();
    let a = dir_a.to_string_lossy().to_string();
    let b = dir_b.to_string_lossy().to_string();
    let list = |path: &str| {
        completion(
            vec![Content::ToolCall {
                id: "l".into(),
                name: "list".into(),
                arguments: serde_json::json!({ "path": path }).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        list(&a),
        list(&b),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent.run_turn("inspect the dirs", &mut ui).await.unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("same inspection output")),
        "expected result-hash no-progress nudge, got: {:?}",
        ui.statuses
    );
    assert_eq!(
        ui.tool_results
            .iter()
            .filter(|(name, _)| name == "list")
            .count(),
        2,
        "the guard should fire after observing the repeated output"
    );
    assert!(ui.turn_end.is_some(), "turn completed after the nudge");
    let _ = std::fs::remove_dir_all(dir_a);
    let _ = std::fs::remove_dir_all(dir_b);
}

#[tokio::test]
async fn nudges_when_model_re_reads_already_inspected_files_in_a_cycle() {
    // The model reads file A, then file B, then file A again. This is a
    // multi-step read cycle (A→B→A→B→…) that evades the exact-match repeat
    // guard — each round differs from the one right before it — but burns the
    // step budget on large workspaces. The re-read cycle guard catches the
    // third round (re-reading A, already in inspected_paths) and nudges the
    // model to act on the output it already has.
    let path_a = temp_file("reread-cycle-a");
    let path_b = temp_file("reread-cycle-b");
    std::fs::write(&path_a, "fn a() {}\n").unwrap();
    std::fs::write(&path_b, "fn b() {}\n").unwrap();
    let a = path_a.to_string_lossy().to_string();
    let b = path_b.to_string_lossy().to_string();
    let read = |p: &str| {
        completion(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": p }).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        read(&a), // first read of A → executes, A enters inspected_paths
        read(&b), // first read of B → executes, B enters inspected_paths
        read(&a), // re-read of A → first consecutive re-read round, executes
        read(&b), // re-read of B → second consecutive re-read round, caught
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("review the code", &mut ui).await.unwrap();
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("re-read files it already inspected")),
        "expected a re-read cycle nudge, got: {:?}",
        ui.statuses
    );
    // The turn should complete (the model finishes after the nudge), not stall.
    assert!(ui.turn_end.is_some(), "turn completed");
    let _ = std::fs::remove_file(path_a);
    let _ = std::fs::remove_file(path_b);
}

#[tokio::test]
async fn does_not_nudge_mixed_new_and_re_read() {
    // A round that reads one new file alongside one already-inspected file is
    // NOT a re-read cycle — the new file is real progress, so both reads
    // execute and no re-read nudge fires.
    let path_a = temp_file("reread-mixed-a");
    let path_c = temp_file("reread-mixed-c");
    std::fs::write(&path_a, "fn a() {}\n").unwrap();
    std::fs::write(&path_c, "fn c() {}\n").unwrap();
    let a = path_a.to_string_lossy().to_string();
    let c = path_c.to_string_lossy().to_string();
    let read = |p: &str| Content::ToolCall {
        id: "r".into(),
        name: "read".into(),
        arguments: serde_json::json!({ "path": p }).to_string(),
    };
    let responses = vec![
        // Round 1: read A alone (executes, A enters inspected_paths).
        completion(vec![read(&a)], 1, 1),
        // Round 2: read A again AND a new file C in the same round. Not all
        // re-reads → executes both, no re-read nudge.
        completion(vec![read(&a), read(&c)], 1, 1),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("review the code", &mut ui).await.unwrap();
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("re-read files it already inspected")),
        "mixed new + re-read should not trigger the re-read nudge, got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn completed");
    let _ = std::fs::remove_file(path_a);
    let _ = std::fs::remove_file(path_c);
}

#[tokio::test]
async fn read_that_failed_before_write_can_be_retried_after_write() {
    // A missing-file read records a stale inspection signature, but a later
    // write can make the exact same path valid. The cycle guard must allow the
    // post-write read to execute instead of treating it as a pointless re-read.
    let path = temp_file("failed-read-then-write");
    let p = path.to_string_lossy().to_string();
    let read = || {
        completion(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": p.clone() }).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        read(), // missing path -> error, signature is recorded as stale
        write_completion(&p),
        read(), // must execute now that the write created the file
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];

    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent
        .run_turn("create the generated file and inspect it", &mut ui)
        .await
        .unwrap();

    let read_results: Vec<_> = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "read")
        .collect();
    assert_eq!(
        read_results.len(),
        2,
        "the read before and after the write should both execute: {:?}",
        ui.tool_results
    );
    assert!(
        read_results
            .iter()
            .any(|(_, output)| output.contains("Error:")),
        "expected the first missing-file read to surface an error: {:?}",
        read_results
    );
    assert!(
        read_results
            .iter()
            .any(|(_, output)| output.contains("1\tx")),
        "expected the post-write read to return the generated file: {:?}",
        read_results
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn nudges_when_model_re_runs_the_same_searches_in_a_cycle() {
    // A grep cycle (grep "foo" → grep "bar" → grep "foo" → grep "bar") evades
    // the exact-match repeat guard — each round differs from the one before it
    // — but the no-new-evidence guard catches it: the third round re-runs a
    // search already seen, and the fourth is the second consecutive
    // no-new-evidence round, so it fires.
    let grep = |pattern: &str| {
        completion(
            vec![Content::ToolCall {
                id: "g".into(),
                name: "grep".into(),
                arguments: serde_json::json!({ "pattern": pattern, "glob": "*.rs" }).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        grep("foo"), // new → executes, signature seen
        grep("bar"), // new → executes, signature seen
        grep("foo"), // re-run → first no-new-evidence round, executes (grace)
        grep("bar"), // re-run → second consecutive no-new-evidence round, caught
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("review the code", &mut ui).await.unwrap();
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("re-read files it already inspected")),
        "expected a no-new-evidence cycle nudge for the grep cycle, got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn completed");
}

#[tokio::test]
async fn allows_one_re_read_after_new_search_then_catches_the_cycle() {
    // The grace rule: a single re-read right after new evidence (a broader
    // search) is allowed through, but a *second* consecutive no-new-evidence
    // round fires. This mirrors the security-review flow (read X → grep broad
    // → re-read X → re-read X) and proves the guard doesn't suppress a
    // legitimate re-inspection while still catching the cycle.
    let path = temp_file("reread-grace");
    std::fs::write(&path, "fn x() { let y = Some(1).unwrap(); }\n").unwrap();
    let p = path.to_string_lossy().to_string();
    let read = || {
        completion(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": p.clone() }).to_string(),
            }],
            1,
            1,
        )
    };
    let grep = |pattern: &str| {
        completion(
            vec![Content::ToolCall {
                id: "g".into(),
                name: "grep".into(),
                arguments: serde_json::json!({ "pattern": pattern, "glob": "*.rs" }).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        read(),         // read X → new, executes
        grep("unwrap"), // new search → new evidence, executes
        read(),         // re-read X → first no-new-evidence round, grace, executes
        read(),         // re-read X → second consecutive no-new-evidence, caught
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("review the code", &mut ui).await.unwrap();
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("re-read files it already inspected")
                || s.contains("re-ran the same command")),
        "expected the cycle to fire on the second consecutive re-read, got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn completed");
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn stale_nudge_stripped_before_next_turn() {
    // When a turn ends after a repeat-nudge stall, the last message in
    // history is a synthetic user nudge. Without stripping, the next
    // prompt would fold into that nudge via `push_user_or_fold`. This
    // test verifies the nudge is stripped so the next turn starts clean.
    let mut responses = vec![echo_call()];
    // Repeat the same call through the whole repeat-nudge budget so the
    // turn ends with a trailing repeat-nudge.
    for _ in 0..(config().max_repeat_nudges + 1) {
        responses.push(echo_call());
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("check it", &mut ui).await.unwrap();

    // After the turn, the last message should NOT be a nudge (user message
    // with a [hi:nudge:...] marker). It should be the assistant's text or
    // a real user message.
    let msgs = agent.messages();
    let last = msgs.last().expect("history is non-empty");
    if last.role == hi_ai::Role::User {
        let text = last
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(
            !text.starts_with("[hi:nudge:"),
            "trailing nudge should be stripped, but last message is: {text}"
        );
    }
}

#[tokio::test]
async fn next_prompt_does_not_fold_into_stale_nudge() {
    // End-to-end: a turn stalls with a repeat-nudge, then a second turn is
    // sent. The second turn's user message should NOT be folded into the
    // stale nudge — it should be a clean, separate user message. We verify
    // by checking that the model sees the real prompt, not nudge text.
    let mut responses = vec![echo_call()];
    for _ in 0..(config().max_repeat_nudges + 1) {
        responses.push(echo_call());
    }
    // Second turn: a clean text response.
    responses.push(completion(vec![Content::Text("ok".into())], 1, 1));

    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("first task", &mut ui).await.unwrap();

    // Second turn — should start clean, not folded into a nudge.
    let mut ui2 = RecUi::default();
    agent.run_turn("second task", &mut ui2).await.unwrap();

    let msgs = agent.messages();
    // Find the last user message — it should be "second task", not a
    // folded nudge+prompt combination.
    let last_user = msgs
        .iter()
        .rev()
        .find(|m| m.role == hi_ai::Role::User)
        .expect("there is a last user message");
    let text = last_user
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert!(
        !text.contains("[hi:nudge:"),
        "next prompt should not be folded into a stale nudge, got: {text}"
    );
    assert!(
        text.contains("second task"),
        "next prompt should be the real user input, got: {text}"
    );
}

#[tokio::test]
async fn turn_start_strips_stale_nudge_from_resumed_history() {
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let (mut agent, requests) = scripted_agent(
        vec![ProviderStep::Completion(completion(
            vec![Content::Text("ok".into())],
            1,
            1,
        ))],
        config(),
    );
    agent
        .messages_mut()
        .push(Message::user("[hi:nudge:repeat] stale nudge 1"));
    agent
        .messages_mut()
        .push(Message::user("[hi:nudge:continue] stale nudge 2"));
    agent
        .messages_mut()
        .push(Message::user("[hi:nudge:verify] stale nudge 3"));
    agent.persisted = agent.messages().len();
    agent.set_session(Box::new(RecordingSession {
        records: records.clone(),
    }));
    let mut ui = RecUi::default();

    agent.run_turn("new task", &mut ui).await.unwrap();

    agent.messages.validate_for_provider().unwrap();
    let requests = requests.lock().unwrap();
    let sent_text = requests[0]
        .iter()
        .map(|message| message.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !sent_text.contains("[hi:nudge:"),
        "stale synthetic nudge should be stripped before provider request: {sent_text}"
    );
    assert!(
        sent_text.contains("new task"),
        "provider request should contain the real new prompt: {sent_text}"
    );
    assert_eq!(agent.persisted, agent.messages().len());
    assert_eq!(
        records.lock().unwrap().len(),
        1,
        "turn should persist without a stale persisted index"
    );
}

#[tokio::test]
async fn silent_auto_continue_keeps_turn_going_without_status() {
    // The model narrates an announced-but-unperformed next step ("Now let me
    // check the tests.") with no tool call. With max_silent_continues > 0 the
    // agent silently re-prompts it to continue — no status line, no visible
    // nudge — and the model then makes the next tool call and finishes with a
    // recap. The recap ("Done.") is a *finished* answer, not a forward-looking
    // step, so it ends the turn cleanly: no further nudge, no false
    // "incomplete" warning.
    let mut cfg = config();
    cfg.max_silent_continues = 3;
    let responses = vec![
        // Round 1: model makes a tool call (actively working).
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // Round 2: announced next step, no tool call → silent continue.
        completion(
            vec![Content::Text("Now let me check the tests.".into())],
            1,
            1,
        ),
        // Round 3: silently re-prompted, model makes the next tool call.
        completion(
            vec![Content::ToolCall {
                id: "r2".into(),
                name: "read".into(),
                arguments: r#"{"path":"y"}"#.into(),
            }],
            1,
            1,
        ),
        // Round 4: model finishes with a recap → turn ends cleanly.
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("review the code", &mut ui).await.unwrap();
    // The turn completed, consuming exactly the four canned responses — a
    // spurious continue after the "Done." recap would have asked for a fifth
    // and panicked on the empty queue.
    assert!(ui.turn_end.is_some(), "turn completed");
    // No visible "nudging" status during the silent continue, and no false
    // "incomplete" warning — the recap ended the turn cleanly.
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("nudging") || s.contains("incomplete")),
        "silent continue then clean finish: {:?}",
        ui.statuses
    );
    assert_eq!(
        agent.last_turn_telemetry().continue_nudges,
        1,
        "silent continue nudge should be reported in telemetry"
    );
    assert!(
        ui.turn_end
            .as_deref()
            .is_some_and(|summary| summary.contains("1 continue")),
        "turn summary should include continue steering: {:?}",
        ui.turn_end
    );
}

#[tokio::test]
async fn finished_recap_after_tool_use_ends_without_incomplete_warning() {
    // Repro of the reported "review codebase runs a bit, then stops without
    // finishing" bug. A read-only task reads files (tool calls), then gives
    // its final recap as text with no tool call. The recap is a *finished*
    // answer (past tense), not an announced next step, so the turn must end
    // cleanly — no silent-continue nudge, no false "the model kept narrating
    // … may be incomplete" warning. Before the fix, `made_tool_call` alone
    // forced a nudge on any post-tool text, so a finished review churned the
    // whole silent-continue budget and stopped on the warning.
    let mut cfg = config();
    cfg.max_silent_continues = 3;
    let responses = vec![
        // Reads a file (actively working).
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"Cargo.toml"}"#.into(),
            }],
            1,
            1,
        ),
        // Final recap — a finished answer, text only.
        completion(
            vec![Content::Text(
                "I reviewed Cargo.toml. The workspace status is clear and tests pass.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("/review codebase", &mut ui).await.unwrap();
    // The turn ended after exactly the two canned responses — a spurious
    // continue would have asked for a third and panicked on the empty queue.
    assert!(ui.turn_end.is_some(), "turn completed");
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "no false incomplete warning on a finished review: {:?}",
        ui.statuses
    );
    // The recap is the closing message — the turn stopped there rather than
    // churning past it with spurious continues.
    let m = agent.messages();
    assert!(
        m.last().unwrap().text().contains("I reviewed Cargo.toml"),
        "the recap is the model's final response: {:?}",
        m.last().unwrap().text()
    );
}

#[tokio::test]
async fn silent_continue_budget_resets_after_tool_progress() {
    // The actual "review codebase stops without finishing" bug. A long,
    // productive turn that *intermittently* narrates a next step without the
    // tool call (a quirk of some models), but reads a file after each nudge.
    // The silent-continue budget bounds *consecutive* stalls, not their
    // total across the turn: each tool call resets the counter, so the turn
    // keeps going as long as the model makes progress between stalls — even
    // when the number of stalls exceeds max_silent_continues. Before the
    // reset the cumulative counter crept up across the whole turn (stall 1,
    // act, stall 2, act, …) and ended it mid-review with a false "incomplete"
    // warning once the Nth stall hit the budget, despite progress every time.
    let mut cfg = config();
    cfg.max_silent_continues = 1;
    let read = |id: &str, path: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: format!(r#"{{"path":"{path}"}}"#),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // Stall 1: narrates a next step, no tool call → nudge (budget is 1).
        completion(vec![Content::Text("Let me read Cargo.toml.".into())], 1, 1),
        // Recovers: reads a file → must reset the silent-continue counter.
        read("a", "Cargo.toml"),
        // Stall 2: narrates again. With the reset this is still within budget;
        // without it the cumulative counter is already exhausted here.
        completion(vec![Content::Text("Let me read README.md.".into())], 1, 1),
        // Recovers again.
        read("b", "README.md"),
        // Finishes with a recap → clean end.
        completion(
            vec![Content::Text(
                "Reviewed Cargo.toml and README.md. Done.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("/review codebase", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "no false incomplete warning while making progress: {:?}",
        ui.statuses
    );
    // Ran all the way to the recap rather than quitting at the second stall.
    assert!(
        agent.messages().last().unwrap().text().contains("Done."),
        "turn ran to the recap: {:?}",
        agent.messages().last().unwrap().text()
    );
    assert_eq!(
        agent.last_turn_telemetry().continue_nudges,
        2,
        "telemetry should count cumulative continue nudges even though the consecutive budget resets"
    );
}

#[tokio::test]
async fn continue_nudge_forces_tool_choice_on_the_next_round() {
    // When the model narrates instead of acting and gets a silent-continue
    // nudge, the *next* request forces a tool call (tool_mode Required ->
    // tool_choice "required") so the model can't answer the nudge with yet
    // another narration or an empty completion (the observed failure mode of
    // some OpenAI-compat coder models). Once the model acts, the force clears.
    let mut cfg = config();
    cfg.max_silent_continues = 1;
    assert_eq!(cfg.tool_mode, ToolMode::Auto, "precondition: free tool use");
    let responses = vec![
        // R1: narrates a next step, no tool call → nudge + force next round.
        completion(vec![Content::Text("Let me read the code.".into())], 1, 1),
        // R2 (forced): the model calls a tool → force clears.
        completion(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R3: finishes with a recap → turn ends.
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(Box::new(provider), cfg);
    let mut ui = RecUi::default();
    agent.run_turn("review", &mut ui).await.unwrap();
    let modes = modes.lock().unwrap().clone();
    assert_eq!(modes.len(), 3, "three model rounds: {modes:?}");
    assert_eq!(modes[0], ToolMode::Auto, "first round is normal");
    assert_eq!(
        modes[1],
        ToolMode::Required,
        "the round after the nudge forces a tool call"
    );
    assert_eq!(
        modes[2],
        ToolMode::Auto,
        "after the model acted, the force is cleared"
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
async fn zero_max_parallel_tools_is_clamped_instead_of_hanging() {
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
    let mut cfg = config();
    cfg.max_parallel_tools = 0;
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        agent.run_turn("scan", &mut ui),
    )
    .await;

    assert!(result.is_ok(), "zero parallelism should not hang");
    result.unwrap().unwrap();
    assert_eq!(
        agent.last_turn_telemetry().max_concurrent_batch,
        1,
        "zero config should be clamped to serial execution"
    );
    assert_eq!(
        agent.last_turn_telemetry().serial_runs,
        2,
        "both ready reads should run serially under the clamp"
    );
    assert_eq!(ui.tool_results.len(), 2, "both tool calls completed");
}

#[tokio::test]
async fn zero_max_steps_is_clamped_to_one_model_round() {
    let responses = vec![completion(vec![Content::Text("done".into())], 4, 2)];
    let mut cfg = config();
    cfg.max_steps = 0;
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    agent.run_turn("answer once", &mut ui).await.unwrap();

    agent.messages.validate_for_provider().unwrap();
    assert_eq!(
        agent.messages().last().unwrap().text(),
        "done",
        "zero max_steps should not leave a user-only turn"
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("reached step limit (0)")),
        "zero max_steps should be clamped before the cap is reported: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn dynamic_max_steps_apply_only_without_explicit_override() {
    let mut cfg = config();
    cfg.max_steps = 999;
    cfg.max_steps_explicit = false;
    let mut first_agent = agent(
        vec![completion(vec![Content::Text("done".into())], 4, 2)],
        cfg,
    );
    let mut ui = RecUi::default();

    first_agent.run_turn("answer once", &mut ui).await.unwrap();

    assert_eq!(first_agent.last_turn_telemetry().effective_max_steps, 200);

    let inspected_path = temp_file("dynamic-read-only-steps");
    std::fs::write(&inspected_path, "pub fn reviewed() {}\n").unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let mut cfg = config();
    cfg.max_steps = 999;
    cfg.max_steps_explicit = false;
    let mut read_only_agent = agent(
        vec![
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                4,
                2,
            ),
            completion(
                vec![Content::Text(format!(
                    "Findings:\n- `{inspected}` was inspected for this bounded review.\n\nEvidence:\n- Read `{inspected}`.\n\nLimits:\n- Limited to inspected evidence."
                ))],
                4,
                2,
            ),
        ],
        cfg,
    );
    let mut ui = RecUi::default();

    read_only_agent
        .run_turn("/review codebase", &mut ui)
        .await
        .unwrap();

    assert_eq!(
        read_only_agent.last_turn_telemetry().effective_max_steps,
        80
    );
    let _ = std::fs::remove_file(inspected_path);

    let mut cfg = config();
    cfg.max_steps = 7;
    cfg.max_steps_explicit = true;
    let mut second_agent = agent(
        vec![completion(vec![Content::Text("done".into())], 4, 2)],
        cfg,
    );
    let mut ui = RecUi::default();

    second_agent.run_turn("answer once", &mut ui).await.unwrap();

    assert_eq!(second_agent.last_turn_telemetry().effective_max_steps, 7);
}

#[tokio::test]
async fn read_only_review_sprawl_is_bounded() {
    // The "inspection sprawl" failure mode: a read-only review turn reads many
    // *distinct* files (each a new inspection signature, so the repeat/cycle
    // guard never fires) without ever producing findings. Without the sprawl
    // guard this churns until max_steps. The guard should nudge once past the
    // threshold, then force the next model round to answer without tools.
    use crate::steering::REVIEW_INSPECTION_CAP;

    // Create enough real temp files that the reads succeed and count as evidence.
    let n_files = (REVIEW_INSPECTION_CAP + 1) as usize;
    let paths: Vec<String> = (0..n_files)
        .map(|i| {
            let p = temp_file(&format!("sprawl-{i}"));
            std::fs::write(&p, format!("file {i} contents\n")).unwrap();
            p.to_string_lossy().to_string()
        })
        .collect();

    // Each initial round reads a distinct file — never a repeat, always "new
    // evidence". The extra read attempt after the threshold should be blocked
    // by the sprawl guard, then the final response answers from existing
    // evidence.
    let mut responses: Vec<Completion> = paths
        .iter()
        .map(|p| {
            completion(
                vec![Content::ToolCall {
                    id: "r".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": p }).to_string(),
                }],
                1,
                1,
            )
        })
        .collect();
    responses.push(completion(
        vec![Content::Text(format!(
            "Findings:\n- {}: Based on the inspected evidence, no major issue is confirmed from this file alone.\n\nEvidence:\n- Reviewed the inspected files gathered in this turn.\n\nLimits:\n- This is limited to inspected evidence and is not a full repository audit.",
            paths[0]
        ))],
        1,
        1,
    ));

    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(Box::new(provider), config());
    let mut ui = RecUi::default();
    agent.run_turn("/review codebase", &mut ui).await.unwrap();

    // The sprawl nudge fired once the threshold was crossed.
    assert!(
        ui.statuses.iter().any(|s| s.contains("without answering")),
        "expected an inspection-sprawl nudge, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.assistant.contains("fallback summary"),
        "sprawl should force an answer attempt before falling back: {}",
        ui.assistant
    );
    assert!(
        ui.assistant.contains("Findings:") && ui.assistant.contains(&paths[0]),
        "expected the forced text answer as the final answer, got: {}",
        ui.assistant
    );
    let modes = modes.lock().unwrap();
    assert_eq!(
        modes.last(),
        Some(&ToolMode::ChatOnly),
        "the post-sprawl answer round should be forced chat-only: {modes:?}"
    );
    assert!(
        ui.turn_end.is_some(),
        "the turn ended rather than churning to max_steps"
    );

    for p in &paths {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn read_only_review_explicit_four_inspection_cap_forces_findings() {
    let n_files = 5usize;
    let paths: Vec<String> = (0..n_files)
        .map(|i| {
            let p = temp_file(&format!("explicit-sprawl-{i}"));
            std::fs::write(&p, format!("file {i} contents\n")).unwrap();
            p.to_string_lossy().to_string()
        })
        .collect();

    let mut responses: Vec<Completion> = paths
        .iter()
        .map(|p| {
            completion(
                vec![Content::ToolCall {
                    id: "r".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": p }).to_string(),
                }],
                1,
                1,
            )
        })
        .collect();
    responses.push(completion(
        vec![Content::Text(format!(
            "Findings:\n- {}: bounded finding from the inspected evidence.",
            paths[0]
        ))],
        1,
        1,
    ));

    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(Box::new(provider), config());
    let mut ui = RecUi::default();
    agent
        .run_turn(
            "Review this codebase for issues related to ipop/coder-balanced API routing or latency. Use at most 4 file inspections. Do not modify files. Return concise findings only; must finish with Findings.",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.assistant.contains("Findings:") && ui.assistant.contains(&paths[0]),
        "expected forced findings, got: {}",
        ui.assistant
    );
    assert_eq!(agent.last_turn_telemetry().file_reads, 4);
    assert_eq!(agent.last_turn_telemetry().targeted_searches, 0);
    assert!(!agent.last_turn_telemetry().hit_step_cap);
    let modes = modes.lock().unwrap();
    assert_eq!(
        modes.last(),
        Some(&ToolMode::ChatOnly),
        "the post-cap answer round should be forced chat-only: {modes:?}"
    );

    for p in &paths {
        let _ = std::fs::remove_file(p);
    }
}
