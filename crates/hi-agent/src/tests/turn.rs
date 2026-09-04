use super::common::*;
use super::*;

struct FailingCheckpointSession;

struct ImmediateAuditProvider {
    ui_observed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for ImmediateAuditProvider {
    async fn stream(
        &self,
        _request: hi_ai::ChatRequest,
        sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<hi_ai::Completion> {
        sink(hi_ai::StreamEvent::WireAudit(Box::new(hi_ai::WireAudit {
            provider: "test".into(),
            accepted: true,
            response_status: Some(200),
            request_body: Some(serde_json::json!({"secret": "never persist"})),
            ..hi_ai::WireAudit::default()
        })));
        assert!(
            self.ui_observed.load(std::sync::atomic::Ordering::Acquire),
            "the provider callback must synchronously reach the UI before the stream returns"
        );
        Ok(completion(
            vec![hi_ai::Content::Text("provider audit delivered".into())],
            1,
            1,
        ))
    }
}

struct ImmediateAuditUi {
    observed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Ui for ImmediateAuditUi {
    fn provider_request(&mut self, audit: &hi_ai::WireAudit) {
        assert_eq!(audit.response_status, Some(200));
        self.observed
            .store(true, std::sync::atomic::Ordering::Release);
    }
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

struct InterruptFirstStartedToolUi {
    interrupt: std::sync::Arc<std::sync::atomic::AtomicBool>,
    target: &'static str,
    on_result: bool,
    fired: bool,
    statuses: Vec<String>,
}

impl Ui for InterruptFirstStartedToolUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_started(&mut self, name: &str, _: &str) {
        if !self.on_result && !self.fired && name == self.target {
            self.fired = true;
            self.interrupt
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, name: &str, _: &str) {
        if self.on_result && !self.fired && name == self.target {
            self.fired = true;
            self.interrupt
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
    fn status(&mut self, status: &str) {
        self.statuses.push(status.to_string());
    }
    fn nudge(&mut self, status: &str) {
        self.statuses.push(status.to_string());
    }
    fn turn_end(&mut self, _: &str) {}
}

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

#[tokio::test]
async fn provider_wire_audit_reaches_ui_inside_the_stream_callback() {
    let observed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let provider = ImmediateAuditProvider {
        ui_observed: observed.clone(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), config()).unwrap();
    let mut ui = ImmediateAuditUi {
        observed: observed.clone(),
    };

    agent
        .run_turn("exercise the provider", &mut ui)
        .await
        .unwrap();

    assert!(observed.load(std::sync::atomic::Ordering::Acquire));
}

#[test]
fn resume_restores_retained_checkpoint_refs() {
    let checkpoints = (0..55).map(|i| format!("{i:040x}")).collect::<Vec<_>>();

    let agent = Agent::resume(
        std::sync::Arc::new(Canned(Mutex::new(Vec::new()))),
        config(),
        vec![Message::system("system")],
        Usage::default(),
        checkpoints,
        None,
        DecisionLog::default(),
    )
    .unwrap();

    assert_eq!(
        agent.checkpoint_count(),
        MAX_CHECKPOINTS,
        "resume keeps the retained checkpoint stack, capped to the undo limit"
    );
}

#[tokio::test]
async fn durable_mode_requires_a_session_sink() {
    let mut cfg = config();
    cfg.execution = crate::ExecutionMode::Durable;
    let mut agent = agent(
        vec![completion(vec![Content::Text("ok".into())], 1, 1)],
        cfg,
    );

    let error = agent
        .run_turn("continue", &mut NullUi)
        .await
        .expect_err("durable execution without storage must fail closed");

    assert!(error.to_string().contains("persisted session"));
}

#[test]
fn live_durable_toggle_requires_persistence_and_can_be_disabled() {
    let mut agent = agent(Vec::new(), config());
    let error = agent
        .set_execution_mode(crate::ExecutionMode::Durable)
        .expect_err("live durable mode must fail without a session sink");
    assert!(error.to_string().contains("saved session"));
    assert_eq!(agent.execution_mode(), crate::ExecutionMode::Ephemeral);

    agent.set_session(Box::new(RecordingSession {
        records: std::sync::Arc::new(Mutex::new(Vec::new())),
    }));
    agent
        .set_execution_mode(crate::ExecutionMode::Durable)
        .expect("saved sessions can be promoted to durable mode");
    assert_eq!(agent.execution_mode(), crate::ExecutionMode::Durable);
    agent
        .set_execution_mode(crate::ExecutionMode::Ephemeral)
        .expect("durable mode can be disabled for later turns");
    assert_eq!(agent.execution_mode(), crate::ExecutionMode::Ephemeral);
}

#[tokio::test]
async fn durable_mode_checkpoints_prompt_and_completed_tool_batches() {
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let mut cfg = config();
    cfg.execution = crate::ExecutionMode::Durable;
    let mut agent = agent(
        vec![
            completion(
                vec![Content::ToolCall {
                    id: "1".into(),
                    name: "bash".into(),
                    arguments: r#"{"command":"echo durable"}"#.into(),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    agent.set_session(Box::new(RecordingSession {
        records: records.clone(),
    }));

    agent.run_turn("run it", &mut NullUi).await.unwrap();

    assert_eq!(
        records.lock().unwrap().len(),
        3,
        "durable execution records the prompt, tool batch, and settled turn"
    );
}

#[tokio::test]
async fn undo_keeps_checkpoint_when_restore_fails() {
    let mut agent = agent(vec![], config());
    agent
        .workspace
        .checkpoints
        .push("not-a-valid-checkpoint".to_string());

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
    let base = std::env::temp_dir().join(format!(
        "hi-agent-undo-session-failure-{}",
        std::process::id()
    ));
    let workspace = base.join("workspace");
    let state = base.join("state");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(workspace.join("value"), "before").unwrap();
    let before = match hi_tools::checkpoint::create_detailed_with_state(&workspace, &state).await {
        hi_tools::checkpoint::CreateResult::Created(id) => id,
        other => panic!("checkpoint failed: {other:?}"),
    };
    std::fs::write(workspace.join("value"), "after").unwrap();
    let after = match hi_tools::checkpoint::create_detailed_with_state(&workspace, &state).await {
        hi_tools::checkpoint::CreateResult::Created(id) => id,
        other => panic!("checkpoint failed: {other:?}"),
    };
    let mut cfg = config();
    cfg.paths.workspace_root = workspace.clone();
    cfg.paths.state_root = state.clone();
    let mut agent = agent(vec![], cfg);
    agent
        .workspace
        .checkpoints
        .push(hi_tools::checkpoint::sealed_reference(&before, &after));
    agent.set_session(Box::new(FailingCheckpointSession));

    let err = agent.undo().await.unwrap_err();

    assert!(format!("{err:#}").contains("disk full"), "{err:#}");
    assert_eq!(
        agent.checkpoint_count(),
        1,
        "checkpoint stack should stay live when the shortened stack cannot be persisted"
    );
    assert_eq!(
        std::fs::read_to_string(workspace.join("value")).unwrap(),
        "after",
        "failed checkpoint-stack persistence must roll the filesystem forward"
    );
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn oversized_generated_tree_no_longer_blocks_strict_checkpointed_edits() {
    let base =
        std::env::temp_dir().join(format!("hi-agent-checkpoint-limit-{}", std::process::id()));
    let workspace = base.join("workspace");
    let state = base.join("state");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(workspace.join("target")).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    let huge = std::fs::File::create(workspace.join("target/cache.bin")).unwrap();
    huge.set_len(512 * 1024 * 1024 + 1).unwrap();
    let write = completion(
        vec![Content::ToolCall {
            id: "write-target".into(),
            name: "write".into(),
            arguments: serde_json::json!({
                "path": "target/new.rs",
                "content": "fn generated() {}\n"
            })
            .to_string(),
        }],
        1,
        1,
    );
    // Artifact-named trees are outside checkpoint scope by policy now, so an
    // oversized target/ no longer breaks checkpoint creation — strict mode
    // keeps its undo promise for everything the ledger tracks and the edit
    // proceeds. (The old behavior denied every edit in any workspace with a
    // large build tree — the live failure this policy change removed.)
    let done = completion(vec![Content::Text("edited".into())], 1, 1);
    let mut cfg = config();
    cfg.paths.workspace_root = workspace.clone();
    cfg.paths.state_root = state;
    cfg.gates.allow_no_checkpoint = false;
    let mut agent = agent(vec![write, done], cfg);

    agent
        .run_turn("write target/new.rs", &mut NullUi)
        .await
        .unwrap();

    assert!(workspace.join("target/new.rs").exists());
    let entry = agent
        .last_turn_telemetry()
        .tool_timeline
        .iter()
        .find(|entry| entry.tool == "write")
        .expect("write timeline entry");
    assert_eq!(entry.status, hi_tools::ToolStatus::Succeeded);
    assert!(entry.effects.mutation_attempted);
    assert!(entry.effects.mutation_applied);
    assert_eq!(agent.last_turn_telemetry().checkpoint_available, Some(true));
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn tools_unavailable_fast_path_resets_state_and_shows_message() {
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let mut cfg = config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    let mut agent = agent(vec![], cfg);
    agent.report.verify = crate::domain::VerifyEvidence::pass(0, String::new());
    agent.workspace.last_changed_files = vec!["old.rs".to_string()];
    agent.report.last_compat_fallbacks = vec!["compat fallback".to_string()];
    agent.report.last_turn_telemetry = TurnTelemetry {
        repeat_nudges: 7,
        no_progress_streak: 7,
        tool_calls: 3,
        ..TurnTelemetry::default()
    };
    agent.goals.last_plan = vec![PlanStep {
        title: "stale step".to_string(),
        status: PlanStatus::Active,
    }];
    agent
        .messages_mut()
        .push(Message::user("[hi:nudge:continue] stale nudge 1"));
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
    assert_eq!(
        agent.last_turn_telemetry(),
        &TurnTelemetry {
            effective_max_steps: u32::MAX,
            ..TurnTelemetry::default()
        },
        "an early rejection must retain the unlimited model-round setting"
    );
    assert_eq!(agent.goals.last_plan[0].title, "stale step");
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
        std::sync::Arc::new(provider),
        config(),
        history,
        Usage::default(),
        Vec::new(),
        None,
        DecisionLog::default(),
    )
    .unwrap();
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
        std::sync::Arc::new(provider),
        config(),
        history,
        Usage::default(),
        Vec::new(),
        None,
        DecisionLog::default(),
    )
    .unwrap();
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
        std::sync::Arc::new(provider),
        config(),
        history,
        Usage::default(),
        Vec::new(),
        None,
        DecisionLog::default(),
    )
    .unwrap();
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
async fn comprehension_question_gets_repository_context() {
    // Regression: "what does this program do" matched no marker in
    // `task_needs_repository_context`, so the turn ran with NO task context
    // index — and a repo-blind model (observed live with two different
    // models) stalled re-posting its plan instead of exploring. Orientation
    // questions about the program/project must carry the repository index.
    let workspace = IsolatedWorkspace::new("comprehension-context");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("src/main.rs"),
        "fn main() { println!(\"hi\"); }\n",
    )
    .unwrap();
    let read_call = ProviderStep::Completion(completion(
        vec![Content::ToolCall {
            id: "r1".into(),
            name: "read".into(),
            arguments: serde_json::json!({"path": "src/main.rs"}).to_string(),
        }],
        1,
        1,
    ));
    let answer = || {
        ProviderStep::Completion(completion(
            vec![Content::Text(
                "src/main.rs is a small CLI that prints hi.".into(),
            )],
            1,
            1,
        ))
    };
    let (mut agent, requests) = scripted_agent(
        vec![read_call, answer(), answer(), answer(), answer()],
        workspace.config(),
    );
    let mut ui = RecUi::default();
    let _ = agent.run_turn("what does this program do", &mut ui).await;

    let requests = requests.lock().unwrap();
    let request_text = requests[0]
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        request_text.contains("# Task context index"),
        "comprehension questions must carry the repository context index; \
         system prompt was: {}",
        &request_text[..request_text.len().min(1500)]
    );
    assert!(
        request_text.contains("src/main.rs"),
        "the index should surface repository files"
    );
}

#[tokio::test]
async fn targeted_named_mutation_skips_repository_context_index() {
    // Named ≤4-file edits already list the files. Injecting the ranked
    // repository index duplicates them and inflates every request.
    let workspace = IsolatedWorkspace::new("named-mutation-no-index");
    std::fs::write(
        workspace.path("driver.py"),
        "def add(a, b):\n    return a - b\n",
    )
    .unwrap();
    std::fs::write(workspace.path("host.py"), "print('host')\n").unwrap();
    let answer = || {
        ProviderStep::Completion(completion(
            vec![Content::Text("Wrote driver.py.".into())],
            1,
            1,
        ))
    };
    let (mut agent, requests) = scripted_agent(
        vec![answer(), answer(), answer(), answer(), answer()],
        workspace.config(),
    );
    let mut ui = RecUi::default();
    let _ = agent
        .run_turn(
            "Write driver.py for the included host.py tool host.\n\
             Do not rewrite host.py or the oracle.\n\
             Do not edit bug/ yourself — only talk to host.py.",
            &mut ui,
        )
        .await;

    let requests = requests.lock().unwrap();
    let request_text = requests[0]
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !request_text.contains("# Task context index"),
        "named-file mutations must skip the repository index; first request was: {}",
        &request_text[..request_text.len().min(1500)]
    );
}

#[tokio::test]
async fn bookkeeping_only_stall_on_mutation_turn_gets_implementation_repair() {
    // Live stall: an implementation turn burned the entire repeat budget on
    // identical update_plan re-posts without ever inspecting or editing. The
    // exhausted-repeat path used to require saw_read/saw_search before handing
    // off to the implementation repair budget, so pure bookkeeping loops fell
    // through to "incomplete · stalled" with zero file changes — exactly the
    // "I started that fix but didn't land the edit" failure. After the fix the
    // turn must convert the stall into an edit nudge, then accept a write and
    // finish without manufacturing a synthetic failure state.
    let workspace = IsolatedWorkspace::new("turn-bookkeeping-impl-repair");
    let plan_args = serde_json::json!({
        "steps": [
            {"title": "Map xAI login in hi", "status": "done"},
            {"title": "Wire web UI approve page", "status": "done"},
            {"title": "Fix review findings", "status": "active"}
        ]
    })
    .to_string();
    let plan_call = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: plan_args.clone(),
            }],
            1,
            1,
        )
    };
    let mut cfg = workspace.config();
    // Keep the default budget (2) so the sequence mirrors production: first
    // execute, then two nudged skips, then a budget-exhausted skip that must
    // become an implementation repair rather than a hard stop.
    cfg.loop_limits.max_repeat_nudges = 2;
    cfg.gates.verification = crate::VerificationMode::Disabled;
    let responses = vec![
        plan_call("plan-1"), // executes
        plan_call("plan-2"), // skip + bookkeeping nudge 1/2
        plan_call("plan-3"), // skip + bookkeeping nudge 2/2
        plan_call("plan-4"), // skip + budget exhausted → impl repair
        write_completion("src/fix.rs"),
        completion(
            vec![Content::Text("Landed the approve-pairing fix.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        .run_turn(
            "fix the approve-pairing auto-approve bug so hi lands the edit",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|s| s
                .contains("implementation burned the bookkeeping-repeat budget without editing")),
        "exhausted bookkeeping-only loop must hand off to implementation repair, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "turn must not hard-stop incomplete after the repair handoff, got: {:?}",
        ui.statuses
    );
    assert!(
        workspace.path("src/fix.rs").exists(),
        "model must be allowed to land the write after the repair nudge"
    );
    let tel = agent.last_turn_telemetry();
    assert_eq!(tel.no_progress_streak, 0);
    agent.messages.validate_for_provider().unwrap();
}

#[tokio::test]
async fn wait_poll_with_changing_output_is_not_repeat_nudged() {
    // The model watches a slow external process by re-running the exact same
    // "sleep && check" command. Each poll returns different output (the
    // process is progressing), so the repeat guard must let every poll
    // execute instead of branding the turn "incomplete · stalled" mid-wait.
    let workspace = IsolatedWorkspace::new("turn-wait-poll-progress");
    let marker = std::env::temp_dir().join(format!("hi-wait-poll-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&marker);
    let poll = || {
        completion(
            vec![Content::ToolCall {
                id: "w".into(),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": format!(
                        "sleep 0; echo tick >> {m}; wc -l < {m}",
                        m = marker.display()
                    )
                })
                .to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        poll(),
        poll(), // exact repeat, but output differs → must execute
        poll(), // again → must execute
        completion(
            vec![Content::Text("Download finished; proceeding.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, workspace.config());
    let mut ui = RecUi::default();
    agent
        .run_turn("wait for the download to finish", &mut ui)
        .await
        .unwrap();
    let _ = std::fs::remove_file(&marker);
    let executed = agent
        .last_turn_telemetry()
        .tool_timeline
        .iter()
        .filter(|entry| entry.tool == "bash")
        .count();
    assert_eq!(
        executed, 3,
        "every changing poll executes: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("re-ran the same command") || s.contains("wait-and-check poll")),
        "no repeat nudges while the poll output changes: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn wait_poll_with_static_output_gets_diagnose_nudge() {
    // The same wait-poll returning byte-identical output means the awaited
    // state stopped changing: the result-hash guard (not the signature guard)
    // nudges the model to diagnose rather than blind-poll, and the turn still
    // ends cleanly once the model reports.
    let workspace = IsolatedWorkspace::new("turn-wait-poll-static");
    let poll = || {
        completion(
            vec![Content::ToolCall {
                id: "w".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "sleep 0; echo waiting"}).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        poll(),
        poll(), // identical output → static-state nudge
        completion(
            vec![Content::Text(
                "The download is stuck at 45 of 76 shards; reported current state.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, workspace.config());
    let mut ui = RecUi::default();
    agent
        .run_turn("wait for the download to finish", &mut ui)
        .await
        .unwrap();
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("wait-and-check poll returned the same output"))
            .count(),
        1,
        "static poll output is nudged once: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().no_progress_streak, 0);
}

#[tokio::test]
async fn repeated_tool_calls_return_a_bounded_provider_error() {
    // The model re-issues the exact same command every round, through the
    // whole repeat-nudge budget: bounded nudges, then a chat-only final-answer
    // recovery. Contentless responses retry under that same policy. If the
    // model still emits tools through the empty-response budget, the turn
    // returns a real unusable-answer error instead of manufacturing a terminal
    // state.
    let mut responses = vec![echo_call()];
    for _ in 0..(config().loop_limits.max_repeat_nudges + 1) {
        responses.push(echo_call()); // exact repeat each round
    }
    for _ in 0..config().loop_limits.max_empty_retries {
        responses.push(echo_call()); // contentless while calls are forbidden
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    let error = agent.run_turn("check it", &mut ui).await.unwrap_err();
    assert!(
        error.to_string().contains("no usable final answer"),
        "unexpected error: {error:#}"
    );
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("re-ran the same command"))
            .count(),
        config().loop_limits.max_repeat_nudges as usize,
        "repeat-nudges are bounded, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("incomplete") || status.contains("stalled")),
        "bounded recovery must not emit a synthetic legacy outcome: {:?}",
        ui.statuses
    );
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
async fn bash_stop_word_cycle_settles_as_typed_no_progress() {
    let mut cfg = config();
    cfg.loop_limits.max_repeat_nudges = 1;
    let responses = vec![
        bash_completion("echo stop"),
        bash_completion("echo quit"),
        bash_completion("echo exit"),
        bash_completion("echo done"),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent.run_turn("stop when complete", &mut ui).await.unwrap();

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
    assert_eq!(outcome.status, crate::TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, crate::TurnStopReason::NoProgress);
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("incomplete") || status.contains("stalled")),
        "the no-progress guard must not manufacture a legacy outcome: {:?}",
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
    assert!(agent.last_turn_telemetry().no_progress_streak > 0);
    assert!(
        ui.assistant
            .contains("could not complete this request after repeated attempts made no progress"),
        "the bounded no-op loop should emit an honest terminal closeout: {}",
        ui.assistant
    );
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

struct RecordScriptedModes {
    scripted: ScriptedProvider,
    modes: std::sync::Arc<Mutex<Vec<ToolMode>>>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for RecordScriptedModes {
    async fn stream(
        &self,
        request: hi_ai::ChatRequest,
        sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<hi_ai::Completion> {
        self.modes.lock().unwrap().push(request.profile.tool_mode);
        hi_ai::Provider::stream(&self.scripted, request, sink).await
    }

    native_tool_test_provider!();
}
#[allow(clippy::type_complexity)]
fn scripted_agent_recording_tool_modes(
    steps: Vec<ProviderStep>,
    cfg: AgentConfig,
) -> (
    Agent,
    std::sync::Arc<Mutex<Vec<Vec<Message>>>>,
    std::sync::Arc<Mutex<Vec<ToolMode>>>,
) {
    let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordScriptedModes {
        scripted: ScriptedProvider {
            steps: Mutex::new(steps),
            requests: requests.clone(),
            max_tokens: None,
        },
        modes: modes.clone(),
    };
    (
        Agent::new(std::sync::Arc::new(provider), cfg).unwrap(),
        requests,
        modes,
    )
}

#[tokio::test]
async fn repeated_no_progress_nudges_force_one_chat_only_final_answer() {
    let mut cfg = config();
    cfg.loop_limits.max_repeat_nudges = 2;
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
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
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

#[tokio::test]
async fn forced_final_malformed_retry_stays_chat_only_without_continue_nudge() {
    let mut cfg = config();
    cfg.loop_limits.max_repeat_nudges = 2;
    cfg.loop_limits.max_empty_retries = 1;
    let steps = vec![
        ProviderStep::Completion(bash_completion("echo stop")),
        ProviderStep::Completion(bash_completion("echo quit")),
        ProviderStep::Completion(bash_completion("echo exit")),
        ProviderStep::Completion(bash_completion("echo done")),
        ProviderStep::Error(hi_ai::ProviderErrorKind::MalformedStream),
        ProviderStep::Completion(completion(
            vec![Content::Text(
                "Stopped after the available no-op output.".into(),
            )],
            1,
            1,
        )),
    ];
    let (mut agent, requests, modes) = scripted_agent_recording_tool_modes(steps, cfg);
    let mut ui = RecUi::default();

    agent.run_turn("stop when complete", &mut ui).await.unwrap();

    let modes = modes.lock().unwrap();
    assert_eq!(
        &modes[modes.len() - 2..],
        &[ToolMode::ChatOnly, ToolMode::ChatOnly],
        "the malformed forced-final request must retry under the same tool-free policy: {modes:?}"
    );
    drop(modes);
    let requests = requests.lock().unwrap();
    let forced_user_text = requests[requests.len() - 2]
        .iter()
        .filter(|message| message.role == Role::User)
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    let retry_user_text = requests[requests.len() - 1]
        .iter()
        .filter(|message| message.role == Role::User)
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        retry_user_text.contains("Stop using tools now"),
        "forced-final instruction was lost on retry: {retry_user_text}"
    );
    assert_eq!(
        forced_user_text, retry_user_text,
        "a malformed forced-final response must not append a conflicting continuation nudge"
    );
    assert!(
        !retry_user_text.contains("previous model response after the tool results was empty"),
        "the tool-continuation nudge contradicted the forced-final instruction: {retry_user_text}"
    );
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 2);
}

#[tokio::test]
async fn forced_final_contentless_retry_stays_chat_only() {
    let mut cfg = config();
    cfg.loop_limits.max_repeat_nudges = 2;
    cfg.loop_limits.max_empty_retries = 1;
    let steps = vec![
        ProviderStep::Completion(bash_completion("echo stop")),
        ProviderStep::Completion(bash_completion("echo quit")),
        ProviderStep::Completion(bash_completion("echo exit")),
        ProviderStep::Completion(bash_completion("echo done")),
        ProviderStep::Completion(completion(Vec::new(), 1, 0)),
        ProviderStep::Completion(completion(
            vec![Content::Text(
                "Stopped after the available no-op output.".into(),
            )],
            1,
            1,
        )),
    ];
    let (mut agent, requests, modes) = scripted_agent_recording_tool_modes(steps, cfg);

    agent
        .run_turn("stop when complete", &mut NullUi)
        .await
        .unwrap();

    let modes = modes.lock().unwrap();
    assert_eq!(
        &modes[modes.len() - 2..],
        &[ToolMode::ChatOnly, ToolMode::ChatOnly],
        "a contentless forced-final completion must retry tool-free: {modes:?}"
    );
    drop(modes);
    let requests = requests.lock().unwrap();
    let retry_user_text = requests
        .last()
        .unwrap()
        .iter()
        .filter(|message| message.role == Role::User)
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(retry_user_text.contains("Stop using tools now"));
    assert!(
        !retry_user_text.contains("previous model response after the tool results was empty"),
        "contentless retry appended tool guidance: {retry_user_text}"
    );
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 2);
}

#[tokio::test]
async fn forced_final_tool_protocol_retry_keeps_only_the_stop_tools_instruction() {
    let mut cfg = config();
    cfg.loop_limits.max_repeat_nudges = 2;
    let steps = vec![
        ProviderStep::Completion(bash_completion("echo stop")),
        ProviderStep::Completion(bash_completion("echo quit")),
        ProviderStep::Completion(bash_completion("echo exit")),
        ProviderStep::Completion(bash_completion("echo done")),
        ProviderStep::Error(hi_ai::ProviderErrorKind::ToolProtocol),
        ProviderStep::Completion(completion(
            vec![Content::Text(
                "Stopped after the available no-op output.".into(),
            )],
            1,
            1,
        )),
    ];
    let (mut agent, requests, modes) = scripted_agent_recording_tool_modes(steps, cfg);

    agent
        .run_turn("stop when complete", &mut NullUi)
        .await
        .unwrap();

    let modes = modes.lock().unwrap();
    assert_eq!(
        &modes[modes.len() - 2..],
        &[ToolMode::ChatOnly, ToolMode::ChatOnly],
        "an invalid forced-final tool turn must retry tool-free: {modes:?}"
    );
    drop(modes);
    let requests = requests.lock().unwrap();
    let forced_user_text = requests[requests.len() - 2]
        .iter()
        .filter(|message| message.role == Role::User)
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    let retry_user_text = requests[requests.len() - 1]
        .iter()
        .filter(|message| message.role == Role::User)
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(retry_user_text.contains("Stop using tools now"));
    assert_eq!(
        forced_user_text, retry_user_text,
        "tool-protocol recovery must not append tool-use guidance to a forced final"
    );
    assert!(
        !retry_user_text.contains("only available tool names"),
        "forced-final retry received contradictory tool guidance: {retry_user_text}"
    );
}

#[tokio::test]
async fn forced_final_survives_provider_context_compaction_and_drop() {
    let mut cfg = config();
    cfg.loop_limits.max_repeat_nudges = 2;
    let steps = vec![
        ProviderStep::Completion(bash_completion("echo stop")),
        ProviderStep::Completion(bash_completion("echo quit")),
        ProviderStep::Completion(bash_completion("echo exit")),
        ProviderStep::Completion(bash_completion("echo done")),
        ProviderStep::RequestTooLarge,
        ProviderStep::RequestTooLarge,
        ProviderStep::Completion(completion(
            vec![Content::Text(
                "Stopped after the available no-op output.".into(),
            )],
            1,
            1,
        )),
    ];
    let (mut agent, requests, modes) = scripted_agent_recording_tool_modes(steps, cfg);
    agent.messages_mut().push(Message::user("older task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "Older task recap.".into(),
        )]));

    agent
        .run_turn("stop when complete", &mut NullUi)
        .await
        .unwrap();

    let modes = modes.lock().unwrap();
    assert_eq!(
        &modes[modes.len() - 3..],
        &[ToolMode::ChatOnly, ToolMode::ChatOnly, ToolMode::ChatOnly],
        "both context-recovery requests must retain the forced-final policy: {modes:?}"
    );
    drop(modes);
    let requests = requests.lock().unwrap();
    let retry_user_text = requests
        .last()
        .unwrap()
        .iter()
        .filter(|message| message.role == Role::User)
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        retry_user_text.contains("Earlier conversation context was omitted"),
        "fixture must exercise the destructive context-drop retry: {retry_user_text}"
    );
    assert!(
        retry_user_text.contains("Stop using tools now"),
        "context drop lost the sticky forced-final instruction: {retry_user_text}"
    );
}

#[tokio::test]
async fn plan_drive_expected_mutation_repeat_never_forces_chat_only() {
    let workspace = IsolatedWorkspace::new("plan-drive-repeat-remains-tool-capable");
    std::fs::write(workspace.path("source.rs"), "fn vote() {}\n").unwrap();
    let source = workspace.path("source.rs").to_string_lossy().to_string();
    let changed = workspace.path("changed.rs").to_string_lossy().to_string();
    let read = || {
        completion(
            vec![Content::ToolCall {
                id: "read-source".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": source}).to_string(),
            }],
            1,
            1,
        )
    };
    let done = completion(
        vec![Content::ToolCall {
            id: "plan-done".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "persist vote transaction code", "status": "done"}]
            })
            .to_string(),
        }],
        1,
        1,
    );
    let mut cfg = workspace.config();
    cfg.gates.allow_unverified = true;
    cfg.loop_limits.max_repeat_nudges = 2;
    let responses = vec![
        read(),
        read(),
        read(),
        read(),
        write_completion(&changed),
        done,
        completion(
            vec![Content::Text("Implemented the plan step.".into())],
            1,
            1,
        ),
    ];
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    agent.restore_plan(vec![hi_tools::PlanStep {
        title: "persist vote transaction code".into(),
        status: hi_tools::PlanStatus::Pending,
    }]);

    agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut NullUi)
        .await
        .unwrap();

    assert!(std::path::Path::new(&changed).exists());
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 0);
    assert!(
        modes
            .lock()
            .unwrap()
            .iter()
            .all(|mode| *mode != ToolMode::ChatOnly),
        "an implementation plan-drive must stay tool-capable through repeat recovery"
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
    fn confirm(&mut self, _: crate::ConfirmationRequest) -> crate::ConfirmationFuture<'_> {
        self.confirm_calls += 1;
        Box::pin(async { crate::ConfirmationResult::Rejected })
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
    cfg.gates.confirm_edits = true;
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
    let denied = agent
        .last_turn_telemetry()
        .tool_timeline
        .iter()
        .find(|entry| entry.tool == "write")
        .expect("denied write timeline entry");
    assert_eq!(denied.status, hi_tools::ToolStatus::Denied);
    assert!(denied.effects.mutation_attempted);
    assert!(!denied.effects.mutation_applied);
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

#[derive(Default)]
struct DenyExternalUi {
    titles: Vec<String>,
    tool_results: Vec<(String, String)>,
}

impl Ui for DenyExternalUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn confirm(&mut self, request: crate::ConfirmationRequest) -> crate::ConfirmationFuture<'_> {
        self.titles.push(request.title().to_string());
        Box::pin(async { crate::ConfirmationResult::Rejected })
    }
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, name: &str, result: &str) {
        self.tool_results
            .push((name.to_string(), result.to_string()));
    }
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

#[tokio::test]
async fn denied_use_tool_does_not_invoke_mcp() {
    let response = completion(
        vec![Content::ToolCall {
            id: "u".into(),
            name: "use_tool".into(),
            arguments: serde_json::json!({
                "server": "github",
                "tool": "create_issue",
                "arguments": { "title": "x" }
            })
            .to_string(),
        }],
        1,
        1,
    );
    let mut cfg = config();
    cfg.gates.confirm_edits = true;
    cfg.memory.offer_mcp = true;
    let mut agent = agent(
        vec![
            response,
            completion(vec![Content::Text("Stopped.".into())], 1, 1),
        ],
        cfg,
    );
    agent.attach_mcp(std::sync::Arc::new(PanicMcp));
    let mut ui = DenyExternalUi::default();
    agent.run_turn("open an issue", &mut ui).await.unwrap();
    assert_eq!(ui.titles, vec!["Confirm MCP tool"]);
    assert!(
        ui.tool_results
            .iter()
            .any(|(name, result)| name == "use_tool" && result.contains("External action skipped")),
        "{:?}",
        ui.tool_results
    );
}

struct PanicMcp;

#[async_trait::async_trait]
impl hi_tools::McpBackend for PanicMcp {
    async fn search(&self, _: Option<&str>) -> anyhow::Result<Vec<hi_tools::McpToolInfo>> {
        panic!("search_tool must not run after a denied confirm");
    }
    async fn call(&self, _: &str, _: &str, _: &serde_json::Value) -> anyhow::Result<String> {
        panic!("use_tool must not invoke MCP after a denied confirm");
    }
}

struct RecordingAdminMcp(std::sync::Arc<std::sync::Mutex<Vec<&'static str>>>);

#[async_trait::async_trait]
impl hi_tools::McpBackend for RecordingAdminMcp {
    async fn search(&self, _: Option<&str>) -> anyhow::Result<Vec<hi_tools::McpToolInfo>> {
        Ok(Vec::new())
    }

    async fn call(&self, _: &str, _: &str, _: &serde_json::Value) -> anyhow::Result<String> {
        unreachable!()
    }

    async fn workspace_admin(&self, _: &str) -> anyhow::Result<String> {
        self.0.lock().unwrap().push("admin");
        Ok("saved".to_string())
    }
}

struct RecordingAdminDurability(std::sync::Arc<std::sync::Mutex<Vec<&'static str>>>);

#[async_trait::async_trait]
impl crate::WorkspaceDurability for RecordingAdminDurability {
    async fn mutation_started(&self, _: Option<Vec<String>>) -> anyhow::Result<()> {
        self.0.lock().unwrap().push("admit");
        Ok(())
    }

    async fn checkpoint(&self) -> anyhow::Result<()> {
        self.0.lock().unwrap().push("checkpoint");
        Ok(())
    }
}

#[tokio::test]
async fn workspace_mcp_admin_is_wrapped_in_the_durability_fence() {
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut subject = agent(Vec::new(), config());
    subject.attach_mcp(std::sync::Arc::new(RecordingAdminMcp(events.clone())));
    subject.set_workspace_durability(Some(std::sync::Arc::new(RecordingAdminDurability(
        events.clone(),
    ))));

    let result = subject
        .mcp_workspace_admin("add docs --http https://example.test")
        .await
        .expect("MCP attached")
        .unwrap();

    assert_eq!(result, "saved");
    assert_eq!(&*events.lock().unwrap(), &["admit", "admin", "checkpoint"]);
}

#[test]
fn remote_rsi_cannot_be_enabled_after_pipefs_controller_is_installed() {
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut subject = agent(Vec::new(), config());
    subject
        .activate_pipefs_workspace_controller("rsi-pipefs-test", 1, false)
        .unwrap();
    subject.set_workspace_durability(Some(std::sync::Arc::new(RecordingAdminDurability(events))));

    let error = subject.set_rsi_enabled(true).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unavailable while PipeFS is active")
    );
}

struct CheckpointBoundarySession(std::sync::Arc<std::sync::Mutex<Vec<Vec<String>>>>);

impl crate::SessionSink for CheckpointBoundarySession {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_checkpoints(&mut self, refs: &[String]) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(refs.to_vec());
        Ok(())
    }
}

#[tokio::test]
async fn workspace_rebind_persists_an_empty_checkpoint_generation_boundary() {
    let boundaries = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut subject = agent(Vec::new(), config());
    subject.workspace.checkpoints = vec!["old-root-checkpoint".to_string()];
    subject.set_session(Box::new(CheckpointBoundarySession(boundaries.clone())));
    let next = tempfile::tempdir().unwrap();
    let next_state = next.path().join("state");

    subject
        .rebind_workspace(next.path(), &next_state)
        .await
        .unwrap();

    assert_eq!(&*boundaries.lock().unwrap(), &[Vec::<String>::new()]);
    assert_eq!(subject.checkpoint_count(), 0);
}

#[tokio::test]
async fn workspace_rebind_accepts_a_preflushed_checkpoint_boundary_without_duplication() {
    let boundaries = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut subject = agent(Vec::new(), config());
    subject.workspace.checkpoints = vec!["portable-root-checkpoint".to_string()];
    subject.set_session(Box::new(CheckpointBoundarySession(boundaries.clone())));
    let next = tempfile::tempdir().unwrap();
    let next_state = next.path().join("state");

    subject.record_workspace_checkpoint_boundary().unwrap();
    // PipeFS flushes the explicit record before disabling the remote root.
    subject
        .rebind_workspace_after_durable_boundary(next.path(), &next_state)
        .await
        .unwrap();

    assert_eq!(&*boundaries.lock().unwrap(), &[Vec::<String>::new()]);
    assert_eq!(subject.checkpoint_count(), 0);
}

#[tokio::test]
async fn denied_browser_exec_does_not_launch() {
    let response = completion(
        vec![Content::ToolCall {
            id: "b".into(),
            name: "browser_exec".into(),
            arguments: serde_json::json!({
                "script": "goto https://example.com\neval document.title"
            })
            .to_string(),
        }],
        1,
        1,
    );
    let mut cfg = config();
    cfg.gates.confirm_edits = true;
    let mut agent = agent(
        vec![
            response,
            completion(vec![Content::Text("Stopped.".into())], 1, 1),
        ],
        cfg,
    );
    agent.set_interactive_session(true);
    let mut ui = DenyExternalUi::default();
    agent
        .run_turn("debug the ui in the browser", &mut ui)
        .await
        .unwrap();
    assert_eq!(ui.titles, vec!["Confirm browser action"]);
    assert!(
        ui.tool_results.iter().any(|(name, result)| {
            name == "browser_exec" && result.contains("External action skipped")
        }),
        "{:?}",
        ui.tool_results
    );
}

#[tokio::test]
async fn denied_mutating_bash_is_retained_as_a_typed_tool_call() {
    let path = temp_file("denied-bash");
    let command = format!("touch '{}'", path.display());
    let response = completion(
        vec![Content::ToolCall {
            id: "b".into(),
            name: "bash".into(),
            arguments: serde_json::json!({ "command": command }).to_string(),
        }],
        1,
        1,
    );
    let mut cfg = config();
    cfg.gates.confirm_edits = true;
    let mut agent = agent(
        vec![
            response,
            completion(vec![Content::Text("Not applied.".into())], 1, 1),
            completion(vec![Content::Text("Not applied.".into())], 1, 1),
            completion(vec![Content::Text("Not applied.".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = DenyEditsUi::default();

    agent.run_turn("change the file", &mut ui).await.unwrap();

    let denied = agent
        .last_turn_telemetry()
        .tool_timeline
        .iter()
        .find(|entry| entry.tool == "bash")
        .expect("denied bash timeline entry");
    assert_eq!(denied.status, hi_tools::ToolStatus::Denied);
    assert!(denied.effects.mutation_attempted);
    assert!(!denied.effects.mutation_applied);
    assert!(!path.exists());
}

#[tokio::test]
async fn interrupted_pending_batch_records_every_typed_cancellation() {
    let path = temp_file("interrupted-batch");
    let sentinel = temp_file("interrupted-batch-sentinel");
    let response = completion(
        vec![
            Content::ToolCall {
                id: "first-write".into(),
                name: "write".into(),
                arguments: serde_json::json!({
                    "path": sentinel.to_string_lossy(),
                    "content": "first"
                })
                .to_string(),
            },
            Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: serde_json::json!({
                    "path": path.to_string_lossy(),
                    "content": "new"
                })
                .to_string(),
            },
            Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": path.to_string_lossy() }).to_string(),
            },
        ],
        1,
        1,
    );
    let mut agent = agent(
        vec![
            response,
            completion(vec![Content::Text("Interrupted.".into())], 1, 1),
        ],
        config(),
    );
    let mut ui = InterruptFirstStartedToolUi {
        interrupt: agent.interrupt_handle(),
        target: "write",
        on_result: false,
        fired: false,
        statuses: Vec::new(),
    };

    agent.run_turn("write the file", &mut ui).await.unwrap();

    let timeline = &agent.last_turn_telemetry().tool_timeline;
    assert_eq!(timeline.len(), 3, "unexpected timeline: {timeline:?}");
    assert!(
        timeline
            .iter()
            .filter(|entry| entry.path == path.to_string_lossy())
            .all(|entry| entry.status == hi_tools::ToolStatus::Cancelled)
    );
    let write = timeline
        .iter()
        .find(|entry| entry.path == path.to_string_lossy())
        .unwrap();
    assert!(write.effects.mutation_attempted);
    assert!(!write.effects.mutation_applied);
    assert!(sentinel.exists());
    assert!(!path.exists());
}

#[tokio::test]
async fn implementation_preflight_consumes_its_interrupt_instead_of_cancelling_next_tool() {
    // Regression for the live failure where Esc was pressed while the hidden
    // implementation-preflight bash was active. The preflight ignored the
    // shared flag, then the model's following update_plan/write consumed the
    // stale signal and was falsely reported as "interrupted by user". That
    // sent the model through no-change repairs until `incomplete · stalled`.
    let workspace = IsolatedWorkspace::new("preflight-interrupt-scope");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"preflight-interrupt-scope\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn old() {}\n").unwrap();

    let responses = vec![
        write_content_completion("src/lib.rs", "pub fn fixed() {}\n"),
        bash_completion("cargo test --quiet"),
        bash_completion("true # validate"),
        completion(
            vec![Content::Text("Implemented and verified.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, workspace.config());
    let mut ui = InterruptFirstStartedToolUi {
        interrupt: agent.interrupt_handle(),
        target: "bash",
        on_result: false,
        fired: false,
        statuses: Vec::new(),
    };

    agent
        .run_turn(
            "Implementation task. You are explicitly allowed and expected to edit files in this disposable workspace, apply patches, and run the verification command. Implement the requested Rust fix in src/lib.rs and verify it.",
            &mut ui,
        )
        .await
        .unwrap();

    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn fixed() {}\n",
        "the interrupt must apply to the visible preflight, not leak to the model's write"
    );
    let timeline = &agent.last_turn_telemetry().tool_timeline;
    assert!(
        timeline
            .iter()
            .any(|entry| entry.tool == "bash" && entry.status == hi_tools::ToolStatus::Cancelled),
        "preflight cancellation should be typed in telemetry: {timeline:?}"
    );
    assert!(
        timeline.iter().any(|entry| {
            entry.tool == "write" && entry.status == hi_tools::ToolStatus::Succeeded
        }),
        "the following write must execute normally: {timeline:?}"
    );
    assert_eq!(agent.last_turn_telemetry().no_progress_streak, 0);
}

#[tokio::test]
async fn late_preflight_interrupt_signal_cannot_cancel_the_models_next_tool() {
    // The TUI may process Esc just after the preflight process exits but before
    // its queued ToolResult clears `current_tool`. Such a signal is too late
    // for the completed preflight and must be discarded at the next batch
    // boundary, never reassigned to the model's first tool.
    let workspace = IsolatedWorkspace::new("late-preflight-interrupt-scope");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"late-preflight-interrupt-scope\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn old() {}\n").unwrap();
    let mut agent = agent(
        vec![
            write_content_completion("src/lib.rs", "pub fn fixed() {}\n"),
            bash_completion("cargo test --quiet"),
            bash_completion("true # validate"),
            completion(
                vec![Content::Text("Implemented and verified.".into())],
                1,
                1,
            ),
        ],
        workspace.config(),
    );
    let mut ui = InterruptFirstStartedToolUi {
        interrupt: agent.interrupt_handle(),
        target: "bash",
        on_result: true,
        fired: false,
        statuses: Vec::new(),
    };

    agent
        .run_turn(
            "Implementation task. You are explicitly allowed and expected to edit files in this disposable workspace, apply patches, and run the verification command. Implement the requested Rust fix in src/lib.rs and verify it.",
            &mut ui,
        )
        .await
        .unwrap();

    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn fixed() {}\n"
    );
    assert!(
        agent
            .last_turn_telemetry()
            .tool_timeline
            .iter()
            .any(|entry| {
                entry.tool == "write" && entry.status == hi_tools::ToolStatus::Succeeded
            })
    );
    assert_eq!(agent.last_turn_telemetry().no_progress_streak, 0);
}

#[tokio::test]
async fn interrupted_bookkeeping_forces_concrete_recovery_round() {
    let plan_call = completion(
        vec![
            Content::ToolCall {
                id: "inspect".into(),
                name: "bash".into(),
                arguments: serde_json::json!({ "command": "pwd" }).to_string(),
            },
            Content::ToolCall {
                id: "plan".into(),
                name: "update_plan".into(),
                arguments: serde_json::json!({
                    "steps": [{"title": "Inspect the project", "status": "active"}]
                })
                .to_string(),
            },
        ],
        1,
        1,
    );
    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(vec![
            plan_call,
            completion(vec![Content::Text("Recovered and finished.".into())], 1, 1),
        ]),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), config()).unwrap();
    let mut ui = InterruptFirstStartedToolUi {
        interrupt: agent.interrupt_handle(),
        target: "bash",
        on_result: false,
        fired: false,
        statuses: Vec::new(),
    };

    agent.run_turn("check the project", &mut ui).await.unwrap();

    let tool_names = tool_names.lock().unwrap();
    let modes = modes.lock().unwrap();
    assert_eq!(modes[1], ToolMode::Required);
    assert!(
        !tool_names[1]
            .iter()
            .any(|name| hi_tools::is_coordination(name)),
        "the recovery round must withhold bookkeeping and demand concrete work: {:?}",
        tool_names[1]
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("continue the active task")),
        "the recovery should be visible: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn confirmation_surfaces_preparation_errors_without_a_blank_prompt_or_reparse() {
    let response = completion(
        vec![Content::ToolCall {
            id: "e".into(),
            name: "edit".into(),
            arguments: r#"{"path":"missing-fields.txt"}"#.into(),
        }],
        1,
        1,
    );
    let mut cfg = config();
    cfg.gates.confirm_edits = true;
    let mut agent = agent(
        vec![
            response,
            completion(vec![Content::Text("The edit was invalid.".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = DenyEditsUi::default();

    agent.run_turn("check it", &mut ui).await.unwrap();

    assert_eq!(
        ui.confirm_calls, 0,
        "an unpreparable mutation must fail before confirmation"
    );
    let edit_result = ui
        .tool_results
        .iter()
        .find(|(name, _)| name == "edit")
        .expect("typed edit failure");
    assert!(edit_result.1.contains("invalid tool arguments"));
    assert!(!edit_result.1.contains("Edit skipped by user"));
}

struct EditDuringConfirmationUi {
    path: std::path::PathBuf,
    preview: Option<String>,
    tool_results: Vec<(String, String)>,
}

impl Ui for EditDuringConfirmationUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn confirm(&mut self, request: crate::ConfirmationRequest) -> crate::ConfirmationFuture<'_> {
        let crate::ConfirmationRequest::FileEdit { diff, .. } = request else {
            panic!("expected file-edit confirmation")
        };
        self.preview = Some(diff);
        // Model an editor save while the confirmation dialog is visible.
        std::fs::write(&self.path, "external editor contents\n").unwrap();
        Box::pin(async { crate::ConfirmationResult::Approved })
    }
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, name: &str, result: &str) {
        self.tool_results
            .push((name.to_string(), result.to_string()));
    }
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

#[tokio::test]
async fn approved_edit_commits_the_previewed_plan_and_refuses_intervening_edits() {
    let path = temp_file("edit-between-preview-and-confirm");
    std::fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();
    let response = completion(
        vec![Content::ToolCall {
            id: "e".into(),
            name: "edit".into(),
            arguments: serde_json::json!({
                "path": path.to_string_lossy(),
                "old_string": "beta",
                "new_string": "BETA"
            })
            .to_string(),
        }],
        1,
        1,
    );
    let mut cfg = config();
    cfg.gates.confirm_edits = true;
    let mut agent = agent(
        vec![
            response,
            completion(
                vec![Content::Text("The edit was not applied.".into())],
                1,
                1,
            ),
        ],
        cfg,
    );
    let mut ui = EditDuringConfirmationUi {
        path: path.as_ref().to_path_buf(),
        preview: None,
        tool_results: Vec::new(),
    };

    agent.run_turn("check it", &mut ui).await.unwrap();

    assert!(
        ui.preview
            .as_deref()
            .is_some_and(|diff| diff.contains("BETA")),
        "missing expected preview; tool results: {:?}",
        ui.tool_results
    );
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "external editor contents\n",
        "approval must not overwrite a file changed after its preview"
    );
    let edit_result = ui
        .tool_results
        .iter()
        .find(|(name, _)| name == "edit")
        .expect("typed edit result");
    assert!(edit_result.1.contains("file changed after preview"));
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn repeated_successful_background_output_poll_is_not_repeat_nudged() {
    let provider = std::sync::Arc::new(Canned(Mutex::new(Vec::new())));
    let mut agent = Agent::new(provider.clone(), config()).unwrap();
    // Two spaced emissions: with the adaptive watcher wait, each defaulted
    // poll parks until its line arrives, so both polls return fresh output
    // while the process is still running.
    let id = agent
        .runtime
        .background()
        .spawn(
            agent.runtime.process_runner(),
            "echo bg-live-one; sleep 0.4; echo bg-live-two; sleep 600",
        )
        .unwrap();
    assert!(id.starts_with("echo-bg-live-one_"), "got: {id}");
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
    provider.0.lock().unwrap().extend(vec![
        bash_output(&id),
        bash_output(&id),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ]);
    let mut ui = RecUi::default();

    agent
        .run_turn("watch the background job", &mut ui)
        .await
        .unwrap();

    let _ = agent.runtime.background().kill(&id);
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
async fn idle_background_output_tight_poll_reports_active_work() {
    // This test exercises the instant-poll steering path; disable the
    // adaptive watcher wait so defaulted polls return immediately.
    let provider = std::sync::Arc::new(Canned(Mutex::new(Vec::new())));
    let mut agent = Agent::new(provider.clone(), config()).unwrap();
    agent.runtime.background().set_poll_wait_base_secs(Some(0));
    let id = agent
        .runtime
        .background()
        .spawn(agent.runtime.process_runner(), "sleep 600")
        .unwrap();
    assert!(id.starts_with("sleep_"), "got: {id}");
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
    // Two free idle polls, then a third that should trip the tight-loop nudge.
    provider.0.lock().unwrap().extend(vec![
        bash_output(&id),
        bash_output(&id),
        bash_output(&id),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ]);
    let mut ui = RecUi::default();

    agent
        .run_turn("watch the quiet background job", &mut ui)
        .await
        .unwrap();

    let _ = agent.runtime.background().kill(&id);
    let bash_output_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "bash_output")
        .count();
    assert_eq!(
        bash_output_results, 3,
        "all three idle polls should execute before the nudge: {:?}",
        ui.tool_results
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("background process is still running")),
        "third consecutive idle poll should trigger an active-work report: {:?}",
        ui.statuses
    );
    assert!(
        ui.tool_results
            .iter()
            .all(|(name, out)| { name != "bash_output" || !out.contains("sleep 600") }),
        "idle polls must not re-echo the command: {:?}",
        ui.tool_results
    );
}

#[tokio::test]
async fn idle_background_poll_budget_exhaustion_reports_progress_without_stalling() {
    // This test exercises the instant-poll steering path; disable the
    // adaptive watcher wait so defaulted polls return immediately.
    let provider = std::sync::Arc::new(Canned(Mutex::new(Vec::new())));
    let mut cfg = config();
    cfg.loop_limits.max_repeat_nudges = 1;
    let mut agent = Agent::new(provider.clone(), cfg).unwrap();
    agent.runtime.background().set_poll_wait_base_secs(Some(0));
    let id = agent
        .runtime
        .background()
        .spawn(agent.runtime.process_runner(), "sleep 600")
        .unwrap();
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
    provider.0.lock().unwrap().extend(vec![
        bash_output(&id),
        bash_output(&id),
        bash_output(&id),
        bash_output(&id),
        completion(
            vec![Content::Text("Download is still running.".into())],
            1,
            1,
        ),
    ]);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("watch the long download", &mut ui)
        .await
        .unwrap();

    let _ = agent.runtime.background().kill(&id);
    assert_ne!(
        outcome.stop_reason,
        crate::TurnStopReason::InfrastructureFailure,
        "statuses={:?}; telemetry={:?}",
        ui.statuses,
        agent.last_turn_telemetry()
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("background process is still running")),
        "expected immediate progress-report recovery: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn waiting_on_live_background_with_fresh_output_ends_with_status_report() {
    // The sol-view failure mode: a download with a progress bar delivers new
    // bytes on every poll, so byte-identical idle detection never fires, and
    // an incomplete plan re-arms the plan-continue nudge after every status
    // answer — 85 nudge/poll cycles in one observed turn. The waiting
    // classifier must key on the process lifecycle and the status answer must
    // end the turn even though the plan still has pending steps.
    let provider = std::sync::Arc::new(Canned(Mutex::new(Vec::new())));
    let mut agent = Agent::new(provider.clone(), config()).unwrap();
    let id = agent
        .runtime
        .background()
        .spawn(
            agent.runtime.process_runner(),
            "i=0; while true; do i=$((i+1)); echo progress-$i; sleep 0.05; done",
        )
        .unwrap();
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
    provider.0.lock().unwrap().extend(vec![
        completion(
            vec![
                Content::ToolCall {
                    id: "plan".into(),
                    name: "update_plan".into(),
                    arguments: serde_json::json!({
                        "steps": [
                            { "title": "Watch the download", "status": "active" },
                            { "title": "Convert the file", "status": "pending" },
                        ]
                    })
                    .to_string(),
                },
                Content::ToolCall {
                    id: "bo0".into(),
                    name: "bash_output".into(),
                    arguments: serde_json::json!({ "id": id }).to_string(),
                },
            ],
            1,
            1,
        ),
        bash_output(&id),
        bash_output(&id),
        completion(
            vec![Content::Text(
                "Work remains in progress: the download is still running; conversion has not started.".into(),
            )],
            1,
            1,
        ),
    ]);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("watch the download and report status", &mut ui)
        .await
        .unwrap();

    let _ = agent.runtime.background().kill(&id);
    assert_ne!(
        outcome.stop_reason,
        crate::TurnStopReason::InfrastructureFailure,
        "statuses={:?}; telemetry={:?}",
        ui.statuses,
        agent.last_turn_telemetry()
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("wait once with wait_secs or wrap up")),
        "the waiting budget should trigger the wrap-up request despite fresh output: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("ending the turn with the status report")),
        "an incomplete plan must not re-arm the continue nudge while waiting: {:?}",
        ui.statuses
    );
    assert!(
        agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("remains in progress"),
        "the status answer is the turn's final message"
    );
    agent.messages.validate_for_provider().unwrap();
}

#[tokio::test]
async fn deliberate_background_process_survives_turn_end() {
    // The sol-view regression: two ~800 GB downloads spawned with
    // run_in_background were reaped by pre-verification turn-end cleanup
    // hours before completion. Work the model deliberately backgrounds must
    // outlive the turn that started it; only auto-backgrounded foreground
    // overruns are turn state.
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "spawn".into(),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": "sleep 600",
                    "run_in_background": true,
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The download is running in the background and will continue after this turn."
                    .into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("start the big download in the background", &mut ui)
        .await
        .unwrap();

    let ids = agent.runtime.background().ids();
    assert_eq!(ids.len(), 1, "the spawned job is registered: {ids:?}");
    assert_eq!(
        agent.runtime.background().outcome(&ids[0]).unwrap().state,
        hi_tools::BackgroundState::Running,
        "a deliberate run_in_background job survives turn end"
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("stopped") && s.contains("background")),
        "turn end must not report reaping deliberate jobs: {:?}",
        ui.statuses
    );
    let _ = agent.runtime.background().kill(&ids[0]);
}

#[tokio::test]
async fn repeated_completed_background_output_poll_is_bounded() {
    // Command-derived handle for `printf bg-complete` (fresh registry → _1).
    let id = "printf-bg-complete_1".to_string();
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
    let started = agent
        .runtime
        .background()
        .spawn(agent.runtime.process_runner(), "printf bg-complete")
        .unwrap();
    assert_eq!(started, id);
    let mut terminal_seen = false;
    for _ in 0..50 {
        let out = agent.runtime.background().poll(&id).unwrap();
        if out.contains(": exited") {
            terminal_seen = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(terminal_seen, "background process should have exited");
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
        bash_output("sh_missing_1"),
        bash_output("sh_missing_2"),
        bash_output("sh_missing_1"),
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
        bash_kill("sh_missing_1"),
        bash_kill("sh_missing_2"),
        bash_kill("sh_missing_1"),
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
async fn missing_background_output_after_prior_mutation_returns_a_bounded_error() {
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
        bash_output("sh_missing_1"),
        bash_output("sh_missing_2"),
    ];
    for i in 0..(config().loop_limits.max_repeat_nudges + 1) {
        responses.push(bash_output(if i % 2 == 0 {
            "sh_missing_1"
        } else {
            "sh_missing_2"
        }));
    }
    for _ in 0..config().loop_limits.max_empty_retries {
        responses.push(bash_output("sh_missing_1"));
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    let error = agent
        .run_turn("fix the harness", &mut ui)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("no usable final answer"),
        "unexpected error: {error:#}"
    );

    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("kept polling stale background process handles"))
            .count(),
        config().loop_limits.max_repeat_nudges as usize,
        "repeat nudges should be bounded, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("incomplete") || status.contains("stalled")),
        "bounded recovery must not emit a synthetic legacy outcome: {:?}",
        ui.statuses
    );
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
async fn missing_background_kill_after_prior_mutation_returns_a_bounded_error() {
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
        bash_kill("sh_missing_1"),
        bash_kill("sh_missing_2"),
    ];
    for i in 0..(config().loop_limits.max_repeat_nudges + 1) {
        responses.push(bash_kill(if i % 2 == 0 {
            "sh_missing_1"
        } else {
            "sh_missing_2"
        }));
    }
    for _ in 0..config().loop_limits.max_empty_retries {
        responses.push(bash_kill("sh_missing_1"));
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    let error = agent
        .run_turn("fix the harness", &mut ui)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("no usable final answer"),
        "unexpected error: {error:#}"
    );

    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("kept using stale background process handles"))
            .count(),
        config().loop_limits.max_repeat_nudges as usize,
        "repeat nudges should be bounded, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("incomplete") || status.contains("stalled")),
        "bounded recovery must not emit a synthetic legacy outcome: {:?}",
        ui.statuses
    );
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
async fn implementation_re_read_exhaustion_settles_as_typed_no_progress() {
    // An implementation task where the model reads a file, then keeps
    // re-reading it through the repeat budget and then ignores the
    // implementation repair nudges — the "explore forever, never edit" failure
    // mode. This is semantic repetition rather than a productive-work count;
    // it must settle as typed no-progress so an autonomous drive cannot
    // immediately restart the same useless cycle.
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
    for _ in 0..(config().loop_limits.max_repeat_nudges + 3) {
        responses.push(read());
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("/build parser implementation", &mut ui)
        .await
        .unwrap();
    assert_eq!(outcome.status, crate::TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, crate::TurnStopReason::NoProgress);
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("incomplete") || status.contains("stalled")),
        "implementation repair exhaustion must not emit a legacy outcome: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 0);
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|status| {
                status.contains("re-read files it already inspected")
                    || status.contains("re-ran the same command")
            })
            .count(),
        config().loop_limits.max_repeat_nudges as usize,
        "implementation repeat nudges stay bounded: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("kept re-running")),
        "should not use the generic stuck-repeating notice for an impl task, got: {:?}",
        ui.statuses
    );
    assert!(
        ui.assistant
            .contains("could not complete this request after repeated attempts made no progress"),
        "implementation exhaustion should emit an honest terminal closeout: {}",
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
    for _ in 0..(config().loop_limits.max_repeat_nudges + 1) {
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
        config().loop_limits.max_repeat_nudges as usize,
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
async fn paging_a_fully_read_file_does_not_claim_the_call_was_identical() {
    // DeepSeek Flash read SPEC.md in full, then asked for later offsets.
    // The skip is correct (the file is complete) but the synthetic result
    // used to say "this call is identical", which is a lie for a different
    // offset and sent the model into arguing with the tool layer.
    let path = temp_file("complete-reread-skip");
    let body: String = (1..=40).map(|n| format!("line {n}\n")).collect();
    std::fs::write(&path, body).unwrap();
    let p = path.to_string_lossy().to_string();
    let read = |id: &str, offset: Option<u64>| {
        let mut args = serde_json::json!({ "path": p.clone() });
        if let Some(offset) = offset {
            args["offset"] = serde_json::json!(offset);
        }
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: args.to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        read("r1", None),
        read("r2", Some(10)),
        read("r3", Some(20)),
        completion(
            vec![Content::Text("Reviewed the requested section.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent
        .run_turn("review Phase 1 in SPEC.md", &mut ui)
        .await
        .unwrap();

    let skipped = agent.messages().iter().find_map(|m| {
        m.content.iter().find_map(|c| match c {
            Content::ToolResult { call_id, output } if call_id == "r3" => Some(output.clone()),
            _ => None,
        })
    });
    let skipped = skipped.expect("third read should have a tool result");
    assert!(
        skipped.contains("returned in full") && skipped.contains("not executed"),
        "completed-file reread skip must be honest, got: {skipped}"
    );
    assert!(
        !skipped.to_ascii_lowercase().contains("identical"),
        "must not claim a different offset is an identical call: {skipped}"
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
        bash_completion("true # validate"),
        completion(
            vec![Content::Text(format!(
                "Changed {write_path_string} and validated with true # validate."
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
    assert_eq!(
        agent.last_turn_telemetry().no_progress_streak,
        0,
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
async fn nudges_when_model_cycles_compound_bash_inspections() {
    // This is the live failure shape from an extended blog-generation turn:
    // alternating `for ...; do head ...; done | sed ...` pages were classified
    // as unknown bash, so A→B→A→B ran forever instead of entering the
    // no-new-evidence guard.
    let output = temp_file("compound-bash-cycle-recovery");
    let page = |range: &str| {
        completion(
            vec![Content::ToolCall {
                id: format!("page-{range}"),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": format!(
                        "for f in Cargo.toml; do head -2 \"$f\"; done | sed -n '{range}'"
                    )
                })
                .to_string(),
            }],
            1,
            1,
        )
    };
    let mut responses = vec![
        page("1p"),
        page("2p"),
        page("1p"), // first no-new-evidence round gets one grace execution
        page("2p"), // second consecutive repeat is skipped and nudged
        write_completion(output.to_string_lossy().as_ref()),
        bash_completion("true # validate"),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    for _ in 0..4 {
        responses.push(completion(vec![Content::Text("Done.".into())], 1, 1));
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("/build the generated blog resource pages", &mut ui)
        .await
        .unwrap();

    assert!(
        ui.statuses.iter().any(|status| {
            status.contains("re-read files it already inspected")
                || status.contains("re-ran the same command")
        }),
        "expected the compound bash cycle to be bounded: {:?}",
        ui.statuses
    );
    assert!(output.exists(), "the model should recover by editing");
    let _ = std::fs::remove_file(output);
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
    for _ in 0..(config().loop_limits.max_repeat_nudges + 1) {
        responses.push(echo_call());
    }
    for _ in 0..config().loop_limits.max_empty_retries {
        responses.push(echo_call());
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    let _ = agent.run_turn("check it", &mut ui).await;

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
    for _ in 0..(config().loop_limits.max_repeat_nudges + 1) {
        responses.push(echo_call());
    }
    for _ in 0..config().loop_limits.max_empty_retries {
        responses.push(echo_call());
    }
    // Second turn: a clean text response.
    responses.push(completion(
        vec![Content::Text(
            "The answer to the second task is four.".into(),
        )],
        1,
        1,
    ));

    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    let _ = agent.run_turn("first task", &mut ui).await;

    // Second turn — should start clean, not folded into a nudge.
    let mut ui2 = RecUi::default();
    agent
        .run_turn("second task: what is two plus two?", &mut ui2)
        .await
        .unwrap();

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
async fn unstructured_forward_phrases_do_not_reenter_a_finished_turn() {
    // Exact regression from a live session: both sentences were part of a
    // terminal recap, but the lexical "I'll" detector injected four hidden
    // continue rounds. Only structured plan/goal state may now auto-continue.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 3;
    let response = "Everything is implemented and tested. Tell me the preferred API shape and I'll adjust it. I'll stop here.";
    let mut agent = agent(
        vec![completion(vec![Content::Text(response.into())], 1, 1)],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("review the code", &mut ui).await.unwrap();

    assert!(ui.turn_end.is_some(), "turn completed");
    assert_eq!(agent.last_turn_telemetry().continue_nudges, 0);
    assert_eq!(agent.messages().last().unwrap().text(), response);
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("incomplete"))
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
    cfg.loop_limits.max_silent_continues = 3;
    let cargo_toml = temp_workspace_path("Cargo.toml");
    std::fs::write(
        cargo_toml.as_ref(),
        "[package]\nname = \"hi-agent-test\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    let cargo_path = cargo_toml.to_string_lossy().to_string();
    let responses = vec![
        // Reads a file (actively working).
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": cargo_path}).to_string(),
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
    let cargo_toml = temp_workspace_path("Cargo.toml");
    std::fs::write(
        cargo_toml.as_ref(),
        "[package]\nname = \"hi-agent\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    let lib_rs = temp_workspace_path("src/lib.rs");
    std::fs::create_dir_all(lib_rs.parent().unwrap()).unwrap();
    std::fs::write(
        lib_rs.as_ref(),
        "//! The agent loop coordinates one turn.\npub fn run() {}\n",
    )
    .unwrap();
    let cargo_path = cargo_toml.to_string_lossy().to_string();
    let lib_path = lib_rs.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![
                Content::ToolCall {
                    id: "1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": cargo_path}).to_string(),
                },
                Content::ToolCall {
                    id: "2".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": lib_path}).to_string(),
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
        "second result is lib.rs: {outputs:?}"
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
    cfg.loop_limits.max_parallel_tools = 0;
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    // This turn reads real workspace files and can contend with the rest of
    // the parallel suite. Keep the bound generous enough to detect the
    // zero-semaphore deadlock this test guards without turning scheduler/disk
    // contention into a false failure.
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
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
    cfg.loop_limits.max_steps = 0;
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
async fn step_cap_folds_wrap_up_nudge_after_recovery_nudge() {
    // A structured plan can remain unfinished on the last normal round. The
    // plan recovery path leaves a synthetic user nudge at the transcript tail; the
    // cap wrap-up must fold into that turn instead of appending consecutive
    // user messages and panicking on the next provider send.
    let responses = vec![
        completion(vec![Content::Text("Let me read the code.".into())], 1, 1),
        completion(
            vec![Content::Text("I could not continue within the cap.".into())],
            1,
            1,
        ),
    ];
    let mut cfg = config();
    cfg.loop_limits.max_steps = 1;
    cfg.loop_limits.max_silent_continues = 1;
    let mut agent = agent(responses, cfg);
    agent.restore_plan(vec![PlanStep {
        title: "read the code".into(),
        status: PlanStatus::Active,
    }]);
    let mut ui = RecUi::default();

    agent.run_turn("continue", &mut ui).await.unwrap();

    agent.messages.validate_for_provider().unwrap();
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("reached step limit (1)")),
        "the capped wrap-up should be visible: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn unlimited_default_and_configured_step_cap_are_honored() {
    // Ordinary turns are unlimited unless the user explicitly sets a cap.
    let mut first_agent = agent(
        vec![completion(vec![Content::Text("done".into())], 4, 2)],
        config(),
    );
    let mut ui = RecUi::default();

    first_agent.run_turn("answer once", &mut ui).await.unwrap();

    assert_eq!(
        first_agent.last_turn_telemetry().effective_max_steps,
        u32::MAX,
        "plain turns have no default model-round cap"
    );

    let inspected_path = temp_file("dynamic-read-only-steps");
    std::fs::write(&inspected_path, "pub fn reviewed() {}\n").unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
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
        config(),
    );
    let mut ui = RecUi::default();

    read_only_agent
        .run_turn("/review codebase", &mut ui)
        .await
        .unwrap();

    assert_eq!(
        read_only_agent.last_turn_telemetry().effective_max_steps,
        u32::MAX,
        "intent classification must not introduce a cap"
    );
    let _ = std::fs::remove_file(inspected_path);

    let mut cfg = config();
    cfg.loop_limits.max_steps = 7;
    let mut second_agent = agent(
        vec![completion(vec![Content::Text("done".into())], 4, 2)],
        cfg,
    );
    let mut ui = RecUi::default();

    second_agent.run_turn("answer once", &mut ui).await.unwrap();

    assert_eq!(second_agent.last_turn_telemetry().effective_max_steps, 7);
}

#[test]
fn automatic_step_setting_restores_unlimited_default() {
    let mut agent = agent(Vec::new(), config());

    assert_eq!(agent.max_steps_setting(), "off");
    agent.set_max_steps_limit(Some(7));
    assert_eq!(agent.max_steps_setting(), "7");
    agent.set_max_steps_auto();
    assert_eq!(agent.max_steps_setting(), "off");
}

#[test]
fn config_snapshot_renders_effective_tool_call_limit() {
    let unlimited = agent(Vec::new(), config());
    assert_eq!(unlimited.max_tool_calls_setting(), "off");
    assert_eq!(unlimited.config_snapshot().max_tool_calls, "off");

    let mut capped = config();
    capped.loop_limits.max_tool_calls = 17;
    let capped = agent(Vec::new(), capped);
    assert_eq!(capped.max_tool_calls_setting(), "17");
    assert_eq!(capped.config_snapshot().max_tool_calls, "17");
}

#[tokio::test]
async fn default_turn_crosses_legacy_32_round_boundary_without_step_cap() {
    const PRODUCTIVE_ROUNDS: u32 = 33;

    let workspace = IsolatedWorkspace::new("unlimited-model-rounds");
    let mut responses = Vec::new();
    for round in 0..PRODUCTIVE_ROUNDS {
        let path = workspace.path(format!("artifact-{round}.txt"));
        responses.push(completion(
            vec![Content::ToolCall {
                id: format!("write-{round}"),
                name: "write".into(),
                arguments: serde_json::json!({
                    "path": path,
                    "content": format!("round {round}\n"),
                })
                .to_string(),
            }],
            1,
            1,
        ));
    }
    responses.push(completion(
        vec![Content::Text(
            "Implemented all 33 requested artifacts.".into(),
        )],
        1,
        1,
    ));

    let mut cfg = workspace.config();
    // This regression isolates the model-round budget. Verification policy is
    // covered separately and must not turn these intentional fixture writes
    // into an unrelated unverified-work failure.
    cfg.gates.allow_unverified = true;
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(
            "Create all 33 requested artifact files, then report completion.",
            &mut ui,
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, crate::TurnStatus::Completed);
    assert_eq!(ui.tool_results.len(), PRODUCTIVE_ROUNDS as usize);
    assert_eq!(agent.last_turn_telemetry().effective_max_steps, u32::MAX);
    assert!(!agent.last_turn_telemetry().hit_step_cap);
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("reached step limit")),
        "the former 32-round default must not stop productive work: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn default_turn_crosses_legacy_48_tool_boundary_without_tool_cap() {
    const PRODUCTIVE_TOOLS: u32 = 49;

    let workspace = IsolatedWorkspace::new("unlimited-tool-calls");
    let mut responses = Vec::new();
    for round in 0..PRODUCTIVE_TOOLS {
        let path = workspace.path(format!("tool-artifact-{round}.txt"));
        responses.push(completion(
            vec![Content::ToolCall {
                id: format!("write-{round}"),
                name: "write".into(),
                arguments: serde_json::json!({
                    "path": path,
                    "content": format!("tool {round}\n"),
                })
                .to_string(),
            }],
            1,
            1,
        ));
    }
    responses.push(completion(
        vec![Content::Text(
            "Implemented all 49 requested artifacts.".into(),
        )],
        1,
        1,
    ));

    let mut cfg = workspace.config();
    cfg.gates.allow_unverified = true;
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(
            "Create all 49 requested artifact files, then report completion.",
            &mut ui,
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, crate::TurnStatus::Completed);
    assert_eq!(ui.tool_results.len(), PRODUCTIVE_TOOLS as usize);
    assert_eq!(agent.max_tool_calls_limit(), u32::MAX);
    assert!(!agent.last_turn_telemetry().hit_step_cap);
    assert!(!agent.last_turn_telemetry().hit_tool_cap);
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("reached tool-call limit")),
        "the former 48-tool default must not stop productive work: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn capped_turn_gets_one_tool_free_wrap_up_round() {
    // Hitting a configured step cap no longer kills the turn mid-flight: the
    // model gets exactly one chat-only round to report where it left the work,
    // and the incomplete turn settles as Failed with a StepLimit diagnostic.
    let mut cfg = config();
    cfg.loop_limits.max_steps = 1;
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(vec![
            bash_completion("echo working"),
            completion(
                vec![Content::Text(
                    "Ran the first check; the remaining verification has not run yet.".into(),
                )],
                1,
                1,
            ),
        ]),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    let outcome = agent.run_turn("run the checks", &mut ui).await.unwrap();

    assert_eq!(outcome.status, crate::TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, crate::TurnStopReason::StepLimit);
    assert!(agent.last_turn_telemetry().hit_step_cap);
    assert!(!agent.last_turn_telemetry().hit_tool_cap);
    assert!(
        ui.assistant.contains("remaining verification"),
        "the wrap-up answer must reach the user: {}",
        ui.assistant
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("asking for a final wrap-up")),
        "expected the wrap-up request status: {:?}",
        ui.statuses
    );
    assert_eq!(
        modes.lock().unwrap().last(),
        Some(&ToolMode::ChatOnly),
        "the wrap-up round must be chat-only"
    );
    agent.messages.validate_for_provider().unwrap();
}

#[tokio::test]
async fn disabled_learning_neither_loads_nor_appends_failure_findings() {
    let workspace = IsolatedWorkspace::new("disabled-failure-learning");
    let ledger = workspace.path(".hi/state/learning/findings.jsonl");
    std::fs::create_dir_all(ledger.parent().unwrap()).unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let finding = crate::learning::Finding {
        ts: now,
        session_id: None,
        turn: None,
        status: TurnStatus::Failed,
        stop_reason: TurnStopReason::StepLimit,
        verification: VerificationStatus::NotApplicable,
        review: ReviewStatus::NotRequired,
        review_unavailable_reason: None,
        last_no_progress_reason: String::new(),
        changed_files: 0,
        model: "test".into(),
        hint_active: None,
        failure_shape: None,
    };
    crate::learning::append_finding(&workspace.path(".hi/state"), &finding);
    crate::learning::append_finding(&workspace.path(".hi/state"), &finding);
    let before = std::fs::read(&ledger).unwrap();

    let mut cfg = workspace.config();
    cfg.memory.learning = false;
    cfg.loop_limits.max_steps = 1;
    let mut agent = agent(
        vec![
            bash_completion("echo working"),
            completion(
                vec![Content::Text("The configured cap stopped the turn.".into())],
                1,
                1,
            ),
        ],
        cfg,
    );
    agent.refresh_memory_context("run the checks");
    assert!(
        agent.task.active_hint_shape.is_none(),
        "disabled learning must not activate a finding-derived steering hint"
    );
    assert!(
        agent
            .task
            .memory_context
            .as_deref()
            .is_none_or(|context| !context.contains("Recent harness findings")),
        "disabled learning leaked finding-derived context: {:?}",
        agent.task.memory_context
    );

    let outcome = agent.run_turn("run the checks", &mut NullUi).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::StepLimit);
    assert_eq!(
        std::fs::read(&ledger).unwrap(),
        before,
        "disabled learning appended a failed-turn finding"
    );
}

#[tokio::test]
async fn simultaneous_step_and_tool_caps_report_both_with_step_precedence() {
    let mut cfg = config();
    cfg.loop_limits.max_steps = 1;
    cfg.loop_limits.max_tool_calls = 1;
    let mut agent = agent(
        vec![
            bash_completion("echo working"),
            completion(
                vec![Content::Text("Stopped at both configured limits.".into())],
                1,
                1,
            ),
        ],
        cfg,
    );
    let mut ui = RecUi::default();

    let outcome = agent.run_turn("run one check", &mut ui).await.unwrap();

    assert_eq!(outcome.status, crate::TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, crate::TurnStopReason::StepLimit);
    assert!(agent.last_turn_telemetry().hit_step_cap);
    assert!(agent.last_turn_telemetry().hit_tool_cap);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("step and tool-call limits")),
        "simultaneous limits should be visible: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn capped_mutating_turn_still_runs_workspace_verification() {
    let workspace = IsolatedWorkspace::new("cap-still-verifies");
    let mut cfg = workspace.config();
    cfg.loop_limits.max_steps = 1;
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let mut agent = agent(
        vec![
            write_content_completion("result.txt", "verified\n"),
            completion(
                vec![Content::Text(
                    "Created result.txt; the configured verifier still needs to run.".into(),
                )],
                1,
                1,
            ),
        ],
        cfg,
    );
    let mut ui = RecUi::default();

    let outcome = agent.run_turn("create result.txt", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::StepLimit);
    assert!(agent.last_turn_telemetry().hit_step_cap);
    assert!(
        agent.report.verify.passed(),
        "cap exit must not skip verification"
    );
    assert_eq!(agent.last_turn_telemetry().verification_executions.len(), 1);
    assert_eq!(
        std::fs::read_to_string(workspace.path("result.txt")).unwrap(),
        "verified\n"
    );
}

#[tokio::test]
async fn read_only_cap_wrap_up_with_answer_is_completed() {
    let path = temp_file("capped-read-only-wrap-up");
    std::fs::write(&path, "bounded evidence\n").unwrap();
    let mut cfg = config();
    cfg.routing.tool_mode = ToolMode::ReadOnly;
    cfg.loop_limits.max_steps = 1;
    let mut agent = agent(
        vec![
            completion(
                vec![Content::ToolCall {
                    id: "read-1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({
                        "path": path.to_string_lossy()
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "The requested evidence was read; no workspace changes were made.".into(),
                )],
                1,
                1,
            ),
        ],
        cfg,
    );
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("Read the file and summarize it.", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.stop_reason, crate::TurnStopReason::StepLimit);
    assert!(agent.last_turn_telemetry().hit_step_cap);
    assert!(ui.assistant.contains("no workspace changes"));
    agent.messages.validate_for_provider().unwrap();
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn capped_turn_wrap_up_round_is_granted_only_once() {
    // If the wrap-up round comes back with tool-call noise instead of text
    // (chat-only requests suppress calls, so they are dropped), the turn ends
    // at the cap rather than granting further rounds.
    let mut cfg = config();
    cfg.loop_limits.max_steps = 1;
    let responses = vec![
        bash_completion("echo working"),
        bash_completion("echo trying to keep going"),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent.run_turn("run the checks", &mut ui).await.unwrap();

    assert_eq!(outcome.status, crate::TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, crate::TurnStopReason::StepLimit);
    assert_eq!(
        ui.tool_results.len(),
        1,
        "the wrap-up round must not execute tools: {:?}",
        ui.tool_results
    );
    agent.messages.validate_for_provider().unwrap();
}

#[tokio::test]
async fn read_only_review_with_explicit_count_language_remains_unbounded() {
    // Count language in a prompt is not an agent-enforced inspection cap. The
    // model may continue gathering distinct evidence until it has enough to
    // answer.
    let explicit_cap = 8u32;
    let n_files = (explicit_cap + 1) as usize;
    let fixtures: Vec<TempTestPath> = (0..n_files)
        .map(|i| {
            let p = temp_file(&format!("sprawl-{i}"));
            std::fs::write(&p, format!("file {i} contents\n")).unwrap();
            p
        })
        .collect();
    let paths: Vec<String> = fixtures
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    // Each initial round reads a distinct file — never a repeat, always "new
    // evidence" — and all of them should execute before the final answer.
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
    let mut agent = Agent::new(std::sync::Arc::new(provider), config()).unwrap();
    let mut ui = RecUi::default();
    let prompt =
        format!("review codebase and discuss status. Use at most {explicit_cap} file inspections.");
    assert!(
        crate::steering::classify_read_only_intent(&prompt).is_none(),
        "this regression must exercise the task-contract structural guard"
    );
    agent.run_turn(&prompt, &mut ui).await.unwrap();

    assert!(
        !ui.assistant.contains("fallback summary"),
        "the review should answer normally: {}",
        ui.assistant
    );
    let answer = agent
        .last_assistant_text()
        .expect("the forced synthesis answer is retained");
    assert!(
        answer.contains("Findings:") && answer.contains(&paths[0]),
        "expected the forced text answer as the final answer, got: {answer}"
    );
    assert_eq!(
        agent.last_turn_telemetry().file_reads,
        n_files as u32,
        "all distinct inspections should run: {:?}",
        ui.statuses
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
async fn read_only_review_crosses_legacy_inspection_count_with_new_evidence() {
    const DISTINCT_READS: usize = 33;
    let fixtures: Vec<TempTestPath> = (0..DISTINCT_READS)
        .map(|i| {
            let path = temp_file(&format!("unlimited-review-{i}"));
            std::fs::write(&path, format!("distinct evidence {i}\n")).unwrap();
            path
        })
        .collect();
    let paths: Vec<String> = fixtures
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect();
    let mut responses = paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            completion(
                vec![Content::ToolCall {
                    id: format!("read-{index}"),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": path}).to_string(),
                }],
                1,
                1,
            )
        })
        .collect::<Vec<_>>();
    responses.push(completion(
        vec![Content::Text(format!(
            "Findings:\n- {}: all 33 distinct evidence files were inspected.\n\nLimits:\n- Findings are limited to the requested review.",
            paths[0]
        ))],
        1,
        1,
    ));

    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("review codebase and discuss status", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert_eq!(
        agent.last_turn_telemetry().file_reads,
        DISTINCT_READS as u32,
        "statuses={:?}; assistant={:?}",
        ui.statuses,
        agent.last_assistant_text(),
    );
    assert!(
        ui.statuses.iter().all(|status| {
            !status.contains("inspection cap")
                && !status.contains("inspection sprawl")
                && !status.contains("without answering")
        }),
        "distinct new evidence must not hit a hidden inspection count: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn read_only_review_ignores_explicit_inspection_count_language() {
    let n_files = 5usize;
    let fixtures: Vec<TempTestPath> = (0..n_files)
        .map(|i| {
            let p = temp_file(&format!("explicit-sprawl-{i}"));
            std::fs::write(&p, format!("file {i} contents\n")).unwrap();
            p
        })
        .collect();
    let paths: Vec<String> = fixtures
        .iter()
        .map(|p| p.to_string_lossy().to_string())
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
    let mut agent = Agent::new(std::sync::Arc::new(provider), config()).unwrap();
    let mut ui = RecUi::default();
    agent
        .run_turn(
            "Review this codebase for issues related to ipop/coder-balanced API routing or latency. Use at most 4 file inspections if useful, but continue whenever more evidence is relevant. Do not modify files. Return concise findings only; must finish with Findings.",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.assistant.contains("Findings:") && ui.assistant.contains(&paths[0]),
        "expected findings after all requested inspections, got: {}",
        ui.assistant
    );
    assert_eq!(agent.last_turn_telemetry().file_reads, 5);
    assert_eq!(agent.last_turn_telemetry().targeted_searches, 0);
    assert!(!agent.last_turn_telemetry().hit_step_cap);
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.contains("inspection cap") && !status.contains("sprawl")),
        "inspection count language must not trigger a stop or forced wrap-up: {:?}",
        ui.statuses
    );

    for p in &paths {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn read_only_review_without_codebase_token_can_keep_inspecting() {
    // A review that does not use the word "codebase" is still allowed to
    // inspect all of the distinct files the model requests. An incidental
    // "at most N" phrase must not create a hidden cap.
    let n_files = 5usize;
    let fixtures: Vec<TempTestPath> = (0..n_files)
        .map(|i| {
            let p = temp_file(&format!("403-sprawl-{i}"));
            std::fs::write(&p, format!("file {i} contents\n")).unwrap();
            p
        })
        .collect();
    let paths: Vec<String> = fixtures
        .iter()
        .map(|p| p.to_string_lossy().to_string())
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
            "Findings:\n- {}: startup_notice stays on screen after models 403.\n\nLimits:\n- Inspected the listed files only.",
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
    let mut cfg = config();
    cfg.loop_limits.max_keep_working = 8;
    cfg.loop_limits.max_empty_retries = 0;
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();
    agent
        .run_turn(
            "review we have some kinda issues : models endpoint returned 403 --- and it seems to stayon the screen. Use at most 4 file inspections if useful.",
            &mut ui,
        )
        .await
        .unwrap();
    let answer = agent
        .last_assistant_text()
        .expect("the final answer is retained");
    assert!(
        answer.contains("Findings:"),
        "expected an answer, got {answer:?}; statuses={:?} modes={:?}",
        ui.statuses,
        modes.lock().unwrap(),
    );
    assert_eq!(agent.last_turn_telemetry().file_reads, n_files as u32);
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.contains("inspection cap") && !status.contains("sprawl")),
        "the incidental count phrase must not trigger inspection sprawl: {:?}",
        ui.statuses
    );
    assert!(
        ui.turn_end.is_some(),
        "turn ended rather than keep-working back into inspection"
    );
    for p in &paths {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn repeated_inspection_does_not_reexecute_forever() {
    let n_files = 5usize;
    let fixtures: Vec<TempTestPath> = (0..n_files)
        .map(|i| {
            let p = temp_file(&format!("keep-working-sprawl-{i}"));
            std::fs::write(&p, format!("file {i} contents\n")).unwrap();
            p
        })
        .collect();
    let paths: Vec<String> = fixtures
        .iter()
        .map(|p| p.to_string_lossy().to_string())
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
    // A later repeated inspection is still suppressed by the no-new-evidence
    // guard. This is distinct from the removed count-based cap: new files
    // remain inspectable without limit.
    responses.push(completion(
        vec![Content::ToolCall {
            id: "r".into(),
            name: "read".into(),
            arguments: serde_json::json!({ "path": &paths[0] }).to_string(),
        }],
        1,
        1,
    ));
    responses.push(completion(
        vec![Content::Text(
            "Findings: the repeated inspection added no new evidence; the review is complete."
                .into(),
        )],
        1,
        1,
    ));
    let mut cfg = config();
    cfg.loop_limits.max_keep_working = 8;
    cfg.loop_limits.max_empty_retries = 0;
    let mut agent = Agent::new(std::sync::Arc::new(Canned(Mutex::new(responses))), cfg).unwrap();
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn(
            "review codebase and discuss status. Inspect as much relevant evidence as needed.",
            &mut ui,
        )
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("incomplete") || status.contains("stalled")),
        "wrap-up exhaustion must not emit a synthetic legacy outcome: {:?}",
        ui.statuses
    );
    assert_eq!(
        ui.tool_results.len(),
        n_files + 1,
        "the repeated inspection may be observed once but must not execute forever: {:?}",
        ui.tool_results
    );
    for p in &paths {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn tool_mutation_refreshes_ranked_task_context_before_next_request() {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "hi-context-refresh-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub fn existing_context() {}\n").unwrap();

    let declaration = "pub fn newly_ranked_context_declaration() {}";
    let mut cfg = config();
    cfg.paths.workspace_root = root.clone();
    cfg.paths.state_root = root.join(".hi/state");
    cfg.gates.allow_unverified = true;
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(
                vec![Content::ToolCall {
                    id: "write-new-context".into(),
                    name: "write".into(),
                    arguments: serde_json::json!({
                        "path": "src/refreshed.rs",
                        "content": format!("{declaration}\n"),
                    })
                    .to_string(),
                }],
                2,
                1,
            )),
            ProviderStep::Completion(bash_completion("cargo check --help")),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "Implemented and validated the context declaration.".into(),
                )],
                2,
                1,
            )),
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 2, 1)),
        ],
        cfg,
    );

    agent
        .run_turn(
            "Implement newly ranked context support in src/refreshed.rs",
            &mut NullUi,
        )
        .await
        .unwrap();

    // Mid-turn rounds are append-only for prompt-cache stability: the write
    // must NOT rewrite the system message (or any earlier message) with a
    // refreshed index — the model already sees its own edit's result.
    {
        let requests = requests.lock().unwrap();
        assert!(requests.len() >= 2, "requests: {requests:#?}");
        assert!(
            !requests[0][0].text().contains(declaration),
            "the declaration did not exist for the initial index"
        );
        assert!(
            !requests[1][0].text().contains(declaration),
            "mid-turn rounds must keep the system message byte-stable: {}",
            requests[1][0].text()
        );
        assert!(
            crate::transcript::Transcript::new(requests[1].clone())
                .validate_for_provider()
                .is_ok(),
            "mid-turn requests must preserve transcript roles"
        );
    }

    // The refreshed index lands in the NEXT turn's volatile context block,
    // attached to that turn's user message.
    agent
        .run_turn(
            "continue to implement context support in src/refreshed.rs",
            &mut RecUi::default(),
        )
        .await
        .unwrap();
    let requests = requests.lock().unwrap();
    let next_turn = requests.last().expect("second turn request");
    let user_text = next_turn
        .iter()
        .rev()
        .find(|m| m.role == hi_ai::Role::User)
        .expect("second turn user message")
        .text();
    assert!(
        user_text.contains(declaration),
        "the next turn's context block carries the refreshed declaration: {user_text}"
    );
    assert!(
        !next_turn[0].text().contains(declaration),
        "the stable system message never absorbs the index"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn system_message_stays_byte_stable_and_context_block_is_singular() {
    // The prompt-cache contract: message[0] is byte-identical across every
    // request of a session, and exactly one volatile context block exists in
    // the transcript (the current turn's). Rebuilding the system message per
    // round was observed to hold provider cache hits under 4%.
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(vec![Content::Text("one".into())], 1, 1)),
            ProviderStep::Completion(completion(vec![Content::Text("two".into())], 1, 1)),
        ],
        config(),
    );
    agent.set_goal(Some("keep the exporter fast".into()));
    let mut ui = RecUi::default();

    agent.run_turn("first question", &mut ui).await.unwrap();
    agent.run_turn("second question", &mut ui).await.unwrap();

    let requests = requests.lock().unwrap();
    assert!(requests.len() >= 2, "two turns → two requests");
    let first_system = requests[0][0].text();
    for (i, request) in requests.iter().enumerate() {
        assert_eq!(
            request[0].text(),
            first_system,
            "system message must stay byte-stable (request {i})"
        );
    }
    let last = requests.last().unwrap();
    let block_count: usize = last
        .iter()
        .map(|m| {
            m.text()
                .matches(crate::transcript::CONTEXT_BLOCK_START)
                .count()
        })
        .sum();
    assert_eq!(
        block_count, 1,
        "exactly one context block lives in the transcript"
    );
    let block_message = last
        .iter()
        .find(|m| m.text().contains(crate::transcript::CONTEXT_BLOCK_START))
        .unwrap()
        .text();
    assert!(
        block_message.contains("keep the exporter fast")
            && block_message.contains("second question"),
        "the current turn's user message carries the block: {block_message}"
    );
    assert!(
        !first_system.contains("keep the exporter fast"),
        "goal state never lands in the stable system message"
    );
    // The second turn's only prefix break is the previous turn's context
    // block being stripped — one break, never at the system message.
    let tel = agent.last_turn_telemetry();
    assert!(
        tel.prefix_break_rounds <= 1,
        "at most one prefix break per turn: {tel:?}"
    );
    assert_ne!(
        tel.earliest_prefix_break,
        Some(0),
        "the system message never breaks the prefix: {tel:?}"
    );
}

#[test]
fn matching_stack_skill_lands_in_volatile_context_not_system_prompt() {
    let workspace = IsolatedWorkspace::new("stack-skill-volatile");
    std::fs::write(workspace.path("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
    let mut cfg = workspace.config();
    // The base test config disables injection; this test asserts it, so opt in.
    cfg.memory.inject_stack_skill = true;
    let agent = agent(vec![], cfg);
    let block = agent.volatile_context_block().unwrap_or_default();
    assert!(
        block.contains("# Active stack skill"),
        "matching rust-workspace pack should be in volatile context: {block}"
    );
    assert!(
        block.contains("[Today]") && block.contains("utc_date:"),
        "parent sessions get a UTC date fragment: {block}"
    );
    assert!(
        block.contains("rust-workspace"),
        "pack slug should be named: {block}"
    );
    let system = agent.messages()[0].text();
    assert!(
        !system.contains("# Active stack skill"),
        "skill bodies must not land in the stable system prompt"
    );
}

#[test]
fn review_turn_injects_code_review_skill_and_skips_stack_pack() {
    let workspace = IsolatedWorkspace::new("review-skill-volatile");
    std::fs::write(workspace.path("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
    let mut cfg = workspace.config();
    cfg.memory.inject_stack_skill = true;
    cfg.memory.inject_review_skill = true;
    let mut agent = agent(vec![], cfg);
    let prompt = crate::command::expand_prompt_macro("/review parser").expect("review macro");
    let contract = crate::TaskContract::derive(&prompt, crate::VerificationMode::Disabled);
    agent.task.set_task(Some(prompt), Some(contract));
    let block = agent.volatile_context_block().unwrap_or_default();
    assert!(
        block.contains("# Active review skill (`code-review`)"),
        "review turn should inject code-review: {block}"
    );
    assert!(
        !block.contains("# Active stack skill"),
        "review turn must not follow rust-workspace: {block}"
    );
}

#[test]
fn tool_free_response_turn_injects_no_review_or_stack_skill() {
    let workspace = IsolatedWorkspace::new("tool-free-no-auto-skill");
    std::fs::write(workspace.path("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
    let mut cfg = workspace.config();
    cfg.memory.inject_stack_skill = true;
    cfg.memory.inject_review_skill = true;
    let mut agent = agent(vec![], cfg);
    let prompt =
        "Without changing files or using tools, answer only: live read-only canary complete.";
    let contract = crate::TaskContract::derive(prompt, crate::VerificationMode::Disabled);
    agent
        .task
        .set_task(Some(prompt.to_string()), Some(contract));

    let block = agent.volatile_context_block().unwrap_or_default();
    assert!(
        !block.contains("# Active review skill"),
        "tool-free response must not inherit code-review instructions: {block}"
    );
    assert!(
        !block.contains("# Active stack skill"),
        "tool-free response must not inherit coding stack instructions: {block}"
    );
}

#[test]
fn explicit_review_with_tools_still_injects_code_review_skill() {
    let workspace = IsolatedWorkspace::new("explicit-review-keeps-skill");
    let mut cfg = workspace.config();
    cfg.memory.inject_review_skill = true;
    let mut agent = agent(vec![], cfg);
    let prompt =
        "Without changing files, review the codebase using tools and report concrete findings.";
    let contract = crate::TaskContract::derive(prompt, crate::VerificationMode::Disabled);
    agent
        .task
        .set_task(Some(prompt.to_string()), Some(contract));

    let block = agent.volatile_context_block().unwrap_or_default();
    assert!(
        block.contains("# Active review skill (`code-review`)"),
        "legitimate review turns must retain the review procedure: {block}"
    );
}

#[test]
fn coding_turn_keeps_stack_pack_without_review_skill() {
    let workspace = IsolatedWorkspace::new("coding-skill-volatile");
    std::fs::write(workspace.path("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
    let mut cfg = workspace.config();
    cfg.memory.inject_stack_skill = true;
    cfg.memory.inject_review_skill = true;
    let mut agent = agent(vec![], cfg);
    let prompt = "Fix the parser in src/lib.rs and add a unit test.";
    let contract = crate::TaskContract::derive(prompt, crate::VerificationMode::Disabled);
    agent
        .task
        .set_task(Some(prompt.to_string()), Some(contract));
    let block = agent.volatile_context_block().unwrap_or_default();
    assert!(
        block.contains("# Active stack skill"),
        "coding turn should keep rust-workspace: {block}"
    );
    assert!(
        !block.contains("# Active review skill"),
        "coding turn must not inject code-review: {block}"
    );
}

#[test]
fn subagent_skips_review_skill() {
    let mut cfg = config();
    cfg.subagents.is_subagent = true;
    cfg.memory.inject_review_skill = true;
    let mut agent = agent(vec![], cfg);
    let prompt = crate::command::expand_prompt_macro("/review parser").expect("review macro");
    let contract = crate::TaskContract::derive(&prompt, crate::VerificationMode::Disabled);
    agent.task.set_task(Some(prompt), Some(contract));
    let block = agent.volatile_context_block().unwrap_or_default();
    assert!(
        !block.contains("# Active review skill"),
        "subagents skip the review pack: {block}"
    );
}

#[test]
fn subagent_volatile_context_omits_today() {
    let mut cfg = config();
    cfg.subagents.is_subagent = true;
    let agent = agent(vec![], cfg);
    let block = agent.volatile_context_block().unwrap_or_default();
    assert!(
        !block.contains("[Today]") && !block.contains("utc_date:"),
        "subagents skip the date fragment: {block}"
    );
}

#[tokio::test]
async fn acceptance_criteria_land_in_volatile_context_after_turn() {
    let workspace = IsolatedWorkspace::new("acceptance-volatile");
    let path = workspace.path("parser.rs");
    let p = path.to_string_lossy().to_string();
    let mut agent = agent(
        vec![
            write_content_completion(&p, "fn parse() -> i32 { 42 }\n"),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        workspace.config(),
    );
    agent
        .run_turn(
            "fix the parser; it must return 42 when the list is empty",
            &mut NullUi,
        )
        .await
        .unwrap();
    let block = agent.volatile_context_block().unwrap_or_default();
    assert!(
        block.contains("# Acceptance criteria"),
        "named must/should bullets should be in volatile context: {block}"
    );
    assert!(
        block.contains("must return 42"),
        "acceptance bullet text missing: {block}"
    );
    assert!(
        !agent.messages()[0].text().contains("# Acceptance criteria"),
        "acceptance text must not land in the stable system prompt"
    );
}

// --- Mid-turn interjection steering --------------------------------------

#[test]
fn interjection_inbox_push_drain_and_ignore_empty() {
    let inbox = crate::InterjectionInbox::default();
    assert!(!inbox.has_pending());
    inbox.push("  "); // whitespace-only is ignored
    inbox.push("focus on the parser");
    inbox.push("and add a test");
    assert!(inbox.has_pending());
    assert_eq!(
        inbox.pending(),
        vec!["focus on the parser", "and add a test"]
    );
    let drained = inbox.drain();
    assert_eq!(drained, vec!["focus on the parser", "and add a test"]);
    assert!(!inbox.has_pending(), "drain empties the queue");
    assert!(inbox.pending().is_empty());
}

/// A message pushed while the turn is running (here, from a Ui hook that fires
/// during the first tool call) is injected as a genuine user message before the
/// next model round — not discarded, not deferred to the next turn.
#[tokio::test]
async fn interjection_is_injected_as_user_message_mid_turn() {
    // Delegates to RecUi, but injects one interjection the first time a tool
    // starts — simulating a message arriving while the turn is running.
    struct InterjectingUi {
        inner: RecUi,
        inbox: crate::InterjectionInbox,
        fired: bool,
    }
    impl Ui for InterjectingUi {
        fn assistant_text(&mut self, text: &str) {
            self.inner.assistant_text(text);
        }
        fn assistant_reasoning(&mut self, text: &str) {
            self.inner.assistant_reasoning(text);
        }
        fn assistant_end(&mut self) {
            self.inner.assistant_end();
        }
        fn tool_call(&mut self, name: &str, arguments: &str) {
            self.inner.tool_call(name, arguments);
        }
        fn tool_result(&mut self, name: &str, result: &str) {
            self.inner.tool_result(name, result);
        }
        fn status(&mut self, text: &str) {
            self.inner.status(text);
        }
        fn turn_end(&mut self, summary: &str) {
            self.inner.turn_end(summary);
        }
        fn tool_started(&mut self, _name: &str, _arguments: &str) {
            if !self.fired {
                self.inbox.push("actually, focus on the parser first");
                self.fired = true;
            }
        }
    }

    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Completion(bash_completion("echo round-one")),
            ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ],
        config(),
    );
    let inbox = agent.interjection_inbox();
    let mut ui = InterjectingUi {
        inner: RecUi::default(),
        inbox,
        fired: false,
    };

    agent.run_turn("start the work", &mut ui).await.unwrap();

    let transcript = agent
        .messages()
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        transcript.contains("The user sent this message while you were working"),
        "interjection framed as a real user message: {transcript}"
    );
    assert!(
        transcript.contains("focus on the parser first"),
        "the user's words are injected: {transcript}"
    );
    assert!(
        ui.inner
            .statuses
            .iter()
            .any(|s| s.contains("received") && s.contains("mid-turn")),
        "the user is told their message landed: {:?}",
        ui.inner.statuses
    );
}

/// A `/btw` question pushed mid-turn is answered off-band: a bounded read-only
/// side completion streams to `btw_answer`, and the main task transcript is left
/// alone (no steering wrapper, no `[user-question]` nudge).
#[tokio::test]
async fn btw_is_answered_off_band_without_transcript_injection() {
    struct BtwUi {
        inner: RecUi,
        inbox: crate::InterjectionInbox,
        fired: bool,
        btw: String,
    }
    impl Ui for BtwUi {
        fn assistant_text(&mut self, text: &str) {
            self.inner.assistant_text(text);
        }
        fn btw_answer(&mut self, text: &str) {
            self.btw.push_str(text);
        }
        fn assistant_reasoning(&mut self, text: &str) {
            self.inner.assistant_reasoning(text);
        }
        fn assistant_end(&mut self) {
            self.inner.assistant_end();
        }
        fn tool_call(&mut self, name: &str, arguments: &str) {
            self.inner.tool_call(name, arguments);
        }
        fn tool_result(&mut self, name: &str, result: &str) {
            self.inner.tool_result(name, result);
        }
        fn status(&mut self, text: &str) {
            self.inner.status(text);
        }
        fn turn_end(&mut self, summary: &str) {
            self.inner.turn_end(summary);
        }
        fn tool_started(&mut self, _name: &str, _arguments: &str) {
            if !self.fired {
                // Simulate the frontend routing `/btw <q>` into the inbox tagged.
                self.inbox.push(format!(
                    "{}{}",
                    crate::BTW_INTERJECTION_PREFIX,
                    // Avoid snapshot-router hits ("working on" → plan fast path).
                    "remind me what color the sky is in the poem?"
                ));
                self.fired = true;
            }
        }
    }

    let (mut agent, requests) = scripted_agent(
        vec![
            // Round 1: tool call (fires the inbox push on tool_started).
            ProviderStep::Completion(bash_completion("echo round-one")),
            // Side-channel `/btw` answer (model path — not a snapshot fast-path).
            ProviderStep::Completion(completion(
                vec![Content::Text("you're finishing round one".into())],
                1,
                1,
            )),
            // Round 2: main task continues.
            ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ],
        config(),
    );
    let inbox = agent.interjection_inbox();
    let mut ui = BtwUi {
        inner: RecUi::default(),
        inbox,
        fired: false,
        btw: String::new(),
    };

    agent.run_turn("start the work", &mut ui).await.unwrap();

    let transcript = agent
        .messages()
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !transcript.contains("asked a side question while you work"),
        "btw must not inject a main-transcript nudge: {transcript}"
    );
    assert!(
        !transcript.contains("remind me what color the sky is"),
        "btw question must not enter the task transcript: {transcript}"
    );
    assert!(
        !transcript.contains("take it into account now"),
        "btw must NOT use the steering wrapper: {transcript}"
    );
    assert!(
        ui.btw.contains("you're finishing round one"),
        "side answer streams to btw_answer, got: {:?}",
        ui.btw
    );
    // Side chrome is pane-only — main status stream must stay clean.
    assert!(
        ui.inner
            .statuses
            .iter()
            .all(|s| !s.contains("❓ btw") && !s.contains("side question")),
        "btw must not spam main statuses: {:?}",
        ui.inner.statuses
    );
    // Side request is a separate provider call whose user message carries the
    // question + session snapshot (and is not folded into the main transcript).
    let reqs = requests.lock().unwrap();
    let side = reqs.iter().find(|msgs| {
        let text = msgs
            .iter()
            .map(Message::text)
            .collect::<Vec<_>>()
            .join("\n");
        text.contains("remind me what color the sky is")
            && text.contains("Current session snapshot:")
    });
    assert!(
        side.is_some(),
        "expected a side completion carrying the /btw question + snapshot; got {} requests",
        reqs.len()
    );
    let side_text = side
        .unwrap()
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        side_text.contains("- model:"),
        "snapshot includes the model line: {side_text}"
    );
    // Main task still completed (third scripted step).
    assert!(
        ui.inner.assistant.contains("done") || transcript.contains("done"),
        "main task should continue after the side answer"
    );
}

/// Main-task `emit_assistant_text` always goes to `assistant_text`. Side answers
/// use `ui.btw_answer` directly from the off-band side completion.
#[test]
fn emit_assistant_text_stays_on_main_stream() {
    #[derive(Default)]
    struct Cap {
        assistant: String,
        btw: String,
    }
    impl Ui for Cap {
        fn assistant_text(&mut self, t: &str) {
            self.assistant.push_str(t);
        }
        fn btw_answer(&mut self, t: &str) {
            self.btw.push_str(t);
        }
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str, _: &str) {}
        fn status(&mut self, _: &str) {}
        fn turn_end(&mut self, _: &str) {}
    }

    let (mut agent, _requests) = scripted_agent(vec![], config());
    let mut ui = Cap::default();

    agent.emit_assistant_text(&mut ui, "task output");
    assert_eq!(ui.assistant, "task output");
    assert!(ui.btw.is_empty(), "main stream must not spill into btw");

    agent.emit_assistant_text(&mut ui, " more");
    assert_eq!(ui.assistant, "task output more");
    assert!(ui.btw.is_empty());
}

/// The `/btw` session snapshot lists live background jobs (id, command, status)
/// so the model can answer "is my job still running / did it finish" without
/// polling. A spawned job appears with its command and a status label.
#[tokio::test]
async fn btw_session_snapshot_includes_background_jobs() {
    let provider = std::sync::Arc::new(Canned(Mutex::new(Vec::new())));
    let agent = Agent::new(provider, config()).unwrap();
    let id = agent
        .runtime
        .background()
        .spawn(agent.runtime.process_runner(), "sleep 30")
        .unwrap();

    let snapshot = agent.btw_session_snapshot();
    assert!(
        snapshot.contains("- background jobs:"),
        "snapshot lists a jobs header: {snapshot}"
    );
    assert!(
        snapshot.contains(&id),
        "snapshot includes the job id {id}: {snapshot}"
    );
    assert!(
        snapshot.contains("sleep 30"),
        "snapshot includes the command: {snapshot}"
    );
    assert!(
        snapshot.contains("(running)"),
        "snapshot shows the running status: {snapshot}"
    );

    let _ = agent.runtime.background().kill(&id);
}

/// Side questions get a session snapshot with cheap git facts (branch, HEAD,
/// first/latest commit). The read-only tool loop can still inspect further;
/// the snapshot covers the common "how old / which branch" asides without a round-trip.
#[tokio::test]
async fn btw_session_snapshot_includes_git_facts() {
    let ws = IsolatedWorkspace::new("btw-git-facts");
    let run = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(ws.path("."))
            .env("GIT_AUTHOR_NAME", "btw")
            .env("GIT_AUTHOR_EMAIL", "btw@example.com")
            .env("GIT_COMMITTER_NAME", "btw")
            .env("GIT_COMMITTER_EMAIL", "btw@example.com")
            .output()
            .expect("git");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    run(&["init", "-q", "-b", "main"]);
    std::fs::write(ws.path("README"), "hi\n").unwrap();
    run(&["add", "README"]);
    run(&[
        "commit",
        "-q",
        "-m",
        "project born",
        "--date=2019-03-04T12:00:00",
    ]);

    let provider = std::sync::Arc::new(Canned(Mutex::new(Vec::new())));
    let agent = Agent::new(provider, ws.config()).unwrap();
    let snapshot = agent.btw_session_snapshot();
    assert!(
        snapshot.contains("- git branch: main"),
        "branch missing: {snapshot}"
    );
    assert!(
        snapshot.contains("- git first commit:") && snapshot.contains("project born"),
        "first commit missing: {snapshot}"
    );
    assert!(
        snapshot.contains("- git latest commit:"),
        "latest commit missing: {snapshot}"
    );
    assert!(
        snapshot.contains("- utc_date:"),
        "utc date missing from /btw snapshot: {snapshot}"
    );
}

/// `/btw` may run a short read-only tool loop (inspect → answer) without
/// injecting anything into the main task transcript — same shape as a mini
/// explore, not a ChatOnly one-shot.
#[tokio::test]
async fn btw_read_only_tool_loop_answers_from_inspection() {
    struct BtwUi {
        inner: RecUi,
        inbox: crate::InterjectionInbox,
        fired: bool,
        btw: String,
        tools: Vec<String>,
    }
    impl Ui for BtwUi {
        fn assistant_text(&mut self, text: &str) {
            self.inner.assistant_text(text);
        }
        fn btw_answer(&mut self, text: &str) {
            self.btw.push_str(text);
        }
        fn btw_tool_result(&mut self, name: &str, _result: &str) {
            self.tools.push(name.to_string());
        }
        fn assistant_reasoning(&mut self, text: &str) {
            self.inner.assistant_reasoning(text);
        }
        fn assistant_end(&mut self) {
            self.inner.assistant_end();
        }
        fn tool_call(&mut self, name: &str, arguments: &str) {
            self.inner.tool_call(name, arguments);
        }
        fn tool_result(&mut self, name: &str, result: &str) {
            self.inner.tool_result(name, result);
        }
        fn status(&mut self, text: &str) {
            self.inner.status(text);
        }
        fn turn_end(&mut self, summary: &str) {
            self.inner.turn_end(summary);
        }
        fn tool_started(&mut self, name: &str, arguments: &str) {
            if !self.fired && name == "bash" {
                self.inbox.push(format!(
                    "{}{}",
                    crate::BTW_INTERJECTION_PREFIX,
                    "what does AGE say?"
                ));
                self.fired = true;
            }
            self.inner.tool_started(name, arguments);
        }
    }

    let ws = IsolatedWorkspace::new("btw-tool-loop");
    std::fs::write(ws.path("AGE.txt"), "born in 2019\n").unwrap();

    let (mut agent, requests) = scripted_agent(
        vec![
            // Round 1: tool call fires the inbox push.
            ProviderStep::Completion(bash_completion("echo main")),
            // Side round 1: inspect AGE.txt.
            ProviderStep::Completion(completion(
                vec![Content::ToolCall {
                    id: "btw_r".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"AGE.txt"}"#.into(),
                }],
                1,
                1,
            )),
            // Side round 2: final answer from the inspection.
            ProviderStep::Completion(completion(
                vec![Content::Text("AGE says born in 2019".into())],
                1,
                1,
            )),
            // Main task continues.
            ProviderStep::Completion(completion(vec![Content::Text("main done".into())], 1, 1)),
        ],
        ws.config(),
    );
    let inbox = agent.interjection_inbox();
    let mut ui = BtwUi {
        inner: RecUi::default(),
        inbox,
        fired: false,
        btw: String::new(),
        tools: Vec::new(),
    };

    agent.run_turn("start the work", &mut ui).await.unwrap();

    assert!(
        ui.btw.contains("born in 2019"),
        "side answer should use the inspection result: {:?}; tools={:?}; statuses={:?}",
        ui.btw,
        ui.tools,
        ui.inner.statuses
    );
    assert!(
        ui.tools.iter().any(|t| t == "read"),
        "btw side loop should have run read: {:?}",
        ui.tools
    );
    let main_text = agent
        .messages()
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !main_text.contains("what does AGE say?"),
        "btw question must stay off the main transcript: {main_text}"
    );
    assert!(
        main_text.contains("main done") || ui.inner.assistant.contains("main done"),
        "main task must still finish: btw={:?} tools={:?} assistant={:?} statuses={:?} messages={main_text}",
        ui.btw,
        ui.tools,
        ui.inner.assistant,
        ui.inner.statuses
    );

    // Side requests include the question + snapshot (tool loop may span 2 provider calls).
    let reqs = requests.lock().unwrap();
    let side_hits = reqs
        .iter()
        .filter(|msgs| {
            let text = msgs
                .iter()
                .map(Message::text)
                .collect::<Vec<_>>()
                .join("\n");
            text.contains("what does AGE say?") && text.contains("Current session snapshot:")
        })
        .count();
    assert!(
        side_hits >= 1,
        "expected side completion(s) for the /btw question; got {} requests",
        reqs.len()
    );
}

struct NativeProgramProvider {
    responses: Mutex<Vec<Completion>>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for NativeProgramProvider {
    async fn stream(
        &self,
        _request: hi_ai::ChatRequest,
        _sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        pop_canned_completion(&self.responses, "NativeProgramProvider")
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        hi_ai::ProviderCapabilities::native_tools(false)
    }
}

#[derive(Default)]
struct SlowProgramConfirmationUi {
    confirmations: usize,
    tool_results: Vec<(String, String)>,
    confirmation_started: std::sync::Arc<tokio::sync::Notify>,
}

impl Ui for SlowProgramConfirmationUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}

    fn confirm(&mut self, _: crate::ConfirmationRequest) -> crate::ConfirmationFuture<'_> {
        self.confirmations += 1;
        let confirmation_started = self.confirmation_started.clone();
        Box::pin(async move {
            confirmation_started.notify_one();
            tokio::time::sleep(std::time::Duration::from_secs(61)).await;
            crate::ConfirmationResult::Rejected
        })
    }

    fn tool_call(&mut self, _: &str, _: &str) {}

    fn tool_result(&mut self, name: &str, result: &str) {
        self.tool_results
            .push((name.to_string(), result.to_string()));
    }

    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

#[tokio::test(start_paused = true)]
async fn productive_program_host_waits_past_the_legacy_total_deadline() {
    let mut cfg = config();
    cfg.program.mode = crate::ProgramMode::Auto;
    cfg.gates.confirm_edits = true;
    let provider = NativeProgramProvider {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "program".into(),
                    name: "run_program".into(),
                    arguments: serde_json::json!({
                        "source": r#"tool("web_fetch", #{url: "https://example.invalid"}); "finished""#
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "The workflow program completed after the host decision.".into(),
                )],
                1,
                1,
            ),
        ]),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    agent.set_permission_mode(crate::PermissionMode::Ask);
    let confirmation_started = std::sync::Arc::new(tokio::sync::Notify::new());
    let mut ui = SlowProgramConfirmationUi {
        confirmation_started: confirmation_started.clone(),
        ..SlowProgramConfirmationUi::default()
    };

    let outcome = {
        let turn = agent.run_turn("run the workflow", &mut ui);
        tokio::pin!(turn);
        tokio::select! {
            () = confirmation_started.notified() => {}
            result = &mut turn => panic!("turn settled before the delayed host decision: {result:?}"),
        }
        tokio::time::advance(std::time::Duration::from_secs(61)).await;
        turn.await.unwrap()
    };

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(ui.confirmations, 1);
    let program_result = ui
        .tool_results
        .iter()
        .find(|(name, _)| name == "run_program")
        .expect("program result was emitted");
    assert!(
        program_result.1.contains(r#""status":"succeeded""#),
        "a slow host decision must not fail the whole program: {program_result:?}"
    );
    assert!(!program_result.1.contains("total time budget"));
}
