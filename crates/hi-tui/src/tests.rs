use super::*;
use crate::app::{review_next_hunk, search_transcript};
use crate::input::HistorySearch;
use ratatui::backend::TestBackend;
use ratatui::style::Color;

mod composer;
mod goal;

fn dump(term: &Terminal<TestBackend>) -> String {
    let buf = term.backend().buffer();
    let mut out = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

fn snapshot_dump(term: &Terminal<TestBackend>) -> String {
    let buf = term.backend().buffer();
    let mut out = String::new();
    for y in 0..buf.area.height {
        let line: String = (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect();
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

fn workflow_snapshot(
    run_id: &str,
    revision: u64,
    status: hi_workflow::WorkflowRunStatus,
) -> hi_workflow::WorkflowRunSnapshot {
    hi_workflow::WorkflowRunSnapshot {
        run_id: run_id.into(),
        revision,
        workflow_name: "deep-research".into(),
        objective: "compare approaches".into(),
        status,
        phases: vec![hi_workflow::WorkflowPhaseSnapshot {
            title: "Research".into(),
            state: "active".into(),
        }],
        current_phase: Some("Research".into()),
        agents: vec![],
        agent_budget: 8,
        agents_used: 2,
        agents_reserved: 0,
        elapsed_ms: 1200,
        pause_message: None,
        result_summary: None,
        history: vec![],
    }
}

#[test]
fn workflow_updates_are_revisioned_and_terminal_updates_are_tombstoned() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::WorkflowUpdated {
        snapshot: workflow_snapshot("run-1", 2, hi_workflow::WorkflowRunStatus::Active),
    });
    app.apply(UiEvent::WorkflowUpdated {
        snapshot: workflow_snapshot("run-1", 1, hi_workflow::WorkflowRunStatus::Failed),
    });
    assert!(app.transcript_text().contains("running"));

    app.apply(UiEvent::WorkflowUpdated {
        snapshot: workflow_snapshot("run-1", 3, hi_workflow::WorkflowRunStatus::Complete),
    });
    app.apply(UiEvent::WorkflowUpdated {
        snapshot: workflow_snapshot("run-1", 4, hi_workflow::WorkflowRunStatus::Active),
    });
    assert!(app.transcript_text().contains("completed"));
    assert!(!app.transcript_text().contains("running"));
    assert_eq!(
        app.transcript
            .iter()
            .filter(|entry| matches!(entry, TranscriptEntry::Workflow { .. }))
            .count(),
        1
    );
}

#[test]
fn completion_reportable_workflow_handoff_is_deduplicated() {
    let mut app = test_app("openai", "gpt-4o");
    let mut snapshot =
        workflow_snapshot("run-handoff", 4, hi_workflow::WorkflowRunStatus::Complete);
    snapshot.result_summary = Some("research finished".into());
    app.apply(UiEvent::WorkflowUpdated {
        snapshot: snapshot.clone(),
    });
    app.apply(UiEvent::WorkflowUpdated { snapshot });
    assert_eq!(app.queue.len(), 1);
    assert!(app.queue[0].contains("research finished"));

    let mut budget = workflow_snapshot(
        "run-budget",
        2,
        hi_workflow::WorkflowRunStatus::BudgetLimited,
    );
    budget.pause_message = Some("raise budget".into());
    app.apply(UiEvent::WorkflowUpdated { snapshot: budget });
    assert_eq!(app.queue.len(), 2);
    assert!(app.queue[1].contains("raise budget"));
}

#[test]
fn terminal_first_workflow_update_creates_durable_block_with_pause_detail() {
    let mut app = test_app("openai", "gpt-4o");
    let mut snapshot = workflow_snapshot("run-2", 1, hi_workflow::WorkflowRunStatus::BudgetLimited);
    snapshot.pause_message = Some("increase the agent budget to resume".into());
    app.apply(UiEvent::WorkflowUpdated { snapshot });

    let text = app.transcript_text();
    assert!(text.contains("budget limited"), "{text}");
    assert!(text.contains("increase the agent budget"), "{text}");
    assert!(text.contains("2/8"), "{text}");
}

#[test]
fn confirmation_modal_renders_mutation_details() {
    let mut app = test_app("openai", "gpt-4o");
    app.confirmation = Some(hi_agent::ConfirmationRequest::ShellMutation {
        command: "rm generated.txt".into(),
        cwd: "/workspace".into(),
    });
    let mut term = Terminal::new(TestBackend::new(120, 20)).unwrap();
    term.draw(|frame| app.render(frame)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("Confirm shell mutation"));
    assert!(screen.contains("rm generated.txt"));
    assert!(screen.contains("y approve"));
    assert!(
        !screen.contains("a always allow"),
        "shell must not offer standing allow: {screen}"
    );
}

#[tokio::test]
async fn channel_confirmation_uses_local_response_channel() {
    use hi_agent::Ui;
    let (tx, _events) = tokio::sync::mpsc::unbounded_channel();
    let (confirmations, mut controls) = tokio::sync::mpsc::unbounded_channel();
    let mut ui = crate::event::ChannelUi {
        tx,
        confirmations,
        event_sink: None,
        approval_store: None,
    };
    let answer = ui.confirm(hi_agent::ConfirmationRequest::FileEdit {
        path: "src/lib.rs".into(),
        diff: "+safe".into(),
    });
    let control = controls.recv().await.unwrap();
    assert!(matches!(
        control.request,
        hi_agent::ConfirmationRequest::FileEdit { .. }
    ));
    control
        .response
        .send(hi_agent::ConfirmationResult::Approved)
        .unwrap();
    assert_eq!(answer.await, hi_agent::ConfirmationResult::Approved);
}

#[tokio::test]
async fn dropped_confirm_receiver_parks_in_approval_store() {
    use hi_policy::{
        ApprovalDecision, ApprovalId, ApprovalRecord, ApprovalState, ApprovalStore,
        CapabilityRequest, OperationDigest,
    };
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    struct MemStore(Mutex<HashMap<String, ApprovalRecord>>);
    impl ApprovalStore for MemStore {
        fn create(&self, request: CapabilityRequest) -> anyhow::Result<ApprovalRecord> {
            let rec = ApprovalRecord {
                request,
                state: ApprovalState::Pending,
                decided_at_ms: None,
                consumed_at_ms: None,
            };
            self.0
                .lock()
                .unwrap()
                .insert(rec.request.approval_id.0.clone(), rec.clone());
            Ok(rec)
        }
        fn get(&self, id: &ApprovalId) -> anyhow::Result<Option<ApprovalRecord>> {
            Ok(self.0.lock().unwrap().get(&id.0).cloned())
        }
        fn decide(
            &self,
            id: &ApprovalId,
            decision: ApprovalDecision,
        ) -> anyhow::Result<ApprovalRecord> {
            let mut map = self.0.lock().unwrap();
            let rec = map
                .get_mut(&id.0)
                .ok_or_else(|| anyhow::anyhow!("missing"))?;
            rec.state = match decision {
                ApprovalDecision::Approved => ApprovalState::Approved,
                _ => ApprovalState::Denied,
            };
            Ok(rec.clone())
        }
        fn claim(
            &self,
            _id: &ApprovalId,
            _digest: &OperationDigest,
        ) -> anyhow::Result<ApprovalRecord> {
            anyhow::bail!("unused")
        }
        fn abandon_run(&self, _run_id: &str) -> anyhow::Result<u64> {
            Ok(0)
        }
        fn pending(&self) -> anyhow::Result<Vec<ApprovalRecord>> {
            Ok(self.0.lock().unwrap().values().cloned().collect())
        }
    }

    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let (confirmations, confirm_rx) = tokio::sync::mpsc::unbounded_channel();
    drop(confirm_rx);
    let store = Arc::new(MemStore(Mutex::new(HashMap::new())));
    let mut ui = crate::event::ChannelUi {
        tx,
        confirmations,
        event_sink: None,
        approval_store: Some(store.clone()),
    };
    use hi_agent::Ui;
    let result = ui
        .confirm(hi_agent::ConfirmationRequest::ShellMutation {
            command: "rm leftover".into(),
            cwd: "/tmp".into(),
        })
        .await;
    assert_eq!(result, hi_agent::ConfirmationResult::Parked);
    let pending = store.pending().unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].state, ApprovalState::Pending);
    assert_eq!(pending[0].request.run_id.as_deref(), Some("parked"));
}

/// A no-op resolver for tests — `/provider` isn't exercised in unit tests.
fn test_resolver() -> ProfileResolver {
    Box::new(|_name| anyhow::bail!("no profiles in tests"))
}

fn test_saver() -> ProfileSaver {
    Box::new(|_form| anyhow::bail!("no profiles in tests"))
}

fn test_loader() -> ProfileLoader {
    Box::new(|_name| anyhow::bail!("no profiles in tests"))
}

fn test_remover() -> ProfileRemover {
    Box::new(|_name| anyhow::bail!("no profiles in tests"))
}

fn test_mlx_switcher() -> MlxProfileSwitcher {
    Box::new(|_run| anyhow::bail!("no mlx profiles in tests"))
}

fn test_local_runtime_switcher() -> LocalRuntimeSwitcher {
    Box::new(|_runtime| anyhow::bail!("no local runtimes in tests"))
}

#[test]
fn selected_model_persists_to_active_profile() {
    let stored = std::sync::Arc::new(std::sync::Mutex::new(ProfileFormData {
        name: "default".into(),
        provider: "pipenetwork".into(),
        api_key: "test-key".into(),
        store_as_env: false,
        model: "pipe/auto-coder".into(),
        base_url: String::new(),
    }));
    let loader_state = stored.clone();
    let saver_state = stored.clone();
    let loader: ProfileLoader = Box::new(move |name| {
        assert_eq!(name, "default");
        Ok(loader_state.lock().unwrap().clone())
    });
    let saver: ProfileSaver = Box::new(move |data| {
        *saver_state.lock().unwrap() = data.clone();
        Ok(vec![ProfileInfo {
            name: data.name.clone(),
            provider: data.provider.clone(),
            model: Some(data.model.clone()),
            base_url: None,
            managed_local_repo: None,
            managed_local_path: None,
        }])
    });

    let mut app = App::new(
        "pipenetwork",
        "pipe/auto-coder",
        vec![ProfileInfo {
            name: "default".into(),
            provider: "pipenetwork".into(),
            model: Some("pipe/auto-coder".into()),
            base_url: None,
            managed_local_repo: None,
            managed_local_path: None,
        }],
        Some("default".into()),
        test_resolver(),
        saver,
        loader,
        test_remover(),
        None,
        test_mlx_switcher(),
        test_local_runtime_switcher(),
        None,
        String::new(),
        None,
        None,
        crate::RaceDefaults::default(),
        None,
    );

    let saved = app
        .persist_active_profile_model("ipop/coder-balanced")
        .expect("persist selected model");

    assert_eq!(saved.as_deref(), Some("default"));
    assert_eq!(
        stored.lock().unwrap().model,
        "ipop/coder-balanced",
        "profile form was rewritten with selected model"
    );
    assert_eq!(
        app.profiles[0].model.as_deref(),
        Some("ipop/coder-balanced")
    );
}

/// `App::new` with empty profiles and dummy callbacks, for tests.
pub(crate) fn test_app(provider: &str, model: &str) -> App {
    let mut app = App::new(
        provider,
        model,
        Vec::new(),
        None,
        test_resolver(),
        test_saver(),
        test_loader(),
        test_remover(),
        None,
        test_mlx_switcher(),
        test_local_runtime_switcher(),
        None,
        String::new(),
        None,
        None,
        crate::RaceDefaults::default(),
        None,
    );
    app.workspace_root = std::path::PathBuf::from("/workspace");
    // Keep snapshots and chrome tests deterministic (no wall-clock AM/PM).
    app.timestamps_enabled = false;
    app
}

#[tokio::test]
async fn sessions_switch_replaces_live_agent_and_ui_session() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");
    app.sync_config = Some(SyncConfig {
        base_url: "http://127.0.0.1:1/v1".into(),
        api_key: "test".into(),
        machine_id: None,
        cwd_digest: None,
    });
    let previous_remote = std::sync::Arc::new(crate::sync_tui::RemoteUi::new(
        crate::sync_tui::SyncConfig {
            base_url: "http://127.0.0.1:1/v1".into(),
            api_key: "test".into(),
        },
        "session-1".into(),
    ));
    app.sync_remote_ui = Some(previous_remote.clone());
    app.push(Line::raw("old transcript"));
    app.session_switcher = Some(Box::new(|id, agent| {
        Box::pin(async move {
            agent
                .apply_loaded_session(
                    vec![
                        hi_ai::Message::system("system"),
                        hi_ai::Message::user("resumed prompt"),
                    ],
                    hi_ai::Usage::default(),
                    Vec::new(),
                    None,
                    hi_agent::DecisionLog::default(),
                    Vec::new(),
                )
                .unwrap();
            Ok(SessionSwitchInfo {
                id: id.to_string(),
                summary: "1 prior message".into(),
            })
        })
    }));

    app.handle_sessions_command(&mut agent, "switch session-2")
        .await;

    assert_eq!(app.sync_session_id.as_deref(), Some("session-2"));
    assert!(!std::sync::Arc::ptr_eq(
        &previous_remote,
        app.sync_remote_ui.as_ref().unwrap()
    ));
    assert!(
        agent
            .messages()
            .iter()
            .any(|m| m.text() == "resumed prompt")
    );
    let transcript = app.transcript_text();
    assert!(transcript.contains("switched to session session-2"));
    assert!(!transcript.contains("old transcript"));
}

#[tokio::test]
async fn sessions_rename_uses_session_manager_callback() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let renamed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let observed = renamed.clone();
    let mut app = test_app("openai", "gpt-4o");
    app.session_lister = Some(Box::new(|| {
        vec![LocalSessionInfo {
            id: "session-2".into(),
            title: "Portal work".into(),
            age: "now".into(),
            lines: 1,
        }]
    }));
    app.session_renamer = Some(Box::new(move |id, name| {
        *observed.lock().unwrap() = Some((id.to_string(), name.to_string()));
        Ok(name.to_string())
    }));

    app.handle_sessions_command(&mut agent, "rename session-2 Portal work")
        .await;

    assert_eq!(
        *renamed.lock().unwrap(),
        Some(("session-2".into(), "Portal work".into()))
    );
    assert!(app.transcript_text().contains("session-2 → Portal work"));
}

#[tokio::test]
async fn sessions_list_uses_one_unified_heading() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");
    app.session_lister = Some(Box::new(|| {
        vec![LocalSessionInfo {
            id: "session-2".into(),
            title: "Portal work".into(),
            age: "now".into(),
            lines: 4,
        }]
    }));

    app.handle_sessions_command(&mut agent, "").await;

    let transcript = app.transcript_text();
    assert!(transcript.contains("sessions (1):"));
    assert!(!transcript.contains("local sessions"));
    assert!(!transcript.contains("remote sessions"));
    assert!(
        transcript.contains("portable")
            || transcript.contains("hosted")
            || transcript.contains("local")
            || transcript.contains("session-2"),
        "list should describe session join mode, got: {transcript}"
    );
}

#[tokio::test]
async fn sessions_attach_resumes_via_switcher_and_replays_history() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");
    app.sync_config = Some(SyncConfig {
        base_url: "http://127.0.0.1:1/v1".into(),
        api_key: "test".into(),
        machine_id: None,
        cwd_digest: None,
    });
    app.session_switcher = Some(Box::new(|id, agent| {
        Box::pin(async move {
            agent
                .apply_loaded_session(
                    vec![
                        hi_ai::Message::system("system"),
                        hi_ai::Message::user("remote prompt from other machine"),
                        hi_ai::Message::assistant(vec![hi_ai::Content::Text(
                            "remote answer".into(),
                        )]),
                    ],
                    hi_ai::Usage::default(),
                    Vec::new(),
                    None,
                    hi_agent::DecisionLog::default(),
                    Vec::new(),
                )
                .unwrap();
            Ok(SessionSwitchInfo {
                id: id.to_string(),
                summary: "2 prior messages".into(),
            })
        })
    }));

    // Force portable continue — without a live control plane the smart path
    // cannot discover host_alive and would error on metadata fetch.
    app.handle_sessions_command(&mut agent, "continue remote-session")
        .await;

    assert_eq!(app.sync_session_id.as_deref(), Some("remote-session"));
    let transcript = app.transcript_text();
    assert!(transcript.contains("switched to session remote-session"));
    assert!(transcript.contains("remote prompt from other machine"));
    assert!(transcript.contains("remote answer"));
    assert!(transcript.contains("remote resume ready"));
}

#[tokio::test]
async fn sessions_host_on_uses_controller_and_drains_remote_queue() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");
    app.sync_session_id = Some("host-session".into());
    app.session_host = Some(Box::new(|enable| {
        Box::pin(async move {
            if enable {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                tx.send("do the thing remotely".into()).unwrap();
                let abort = tokio::spawn(async {}).abort_handle();
                Ok(Some((rx, abort)))
            } else {
                Ok(None)
            }
        })
    }));

    app.handle_sessions_command(&mut agent, "host on").await;
    assert!(app.hosting_remote_input);
    assert!(app.drain_remote_input());
    assert_eq!(
        app.queue.front().map(String::as_str),
        Some("do the thing remotely")
    );

    app.handle_sessions_command(&mut agent, "host off").await;
    assert!(!app.hosting_remote_input);
    assert!(app.remote_input_rx.is_none());
}

#[tokio::test]
async fn remote_drain_respects_prompt_queue_cap() {
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..crate::MAX_PROMPT_QUEUE {
        assert!(app.try_enqueue_prompt(format!("local-{i}")));
    }
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tx.send("remote-overflow-1".into()).unwrap();
    tx.send("remote-overflow-2".into()).unwrap();
    app.remote_input_rx = Some(rx);

    assert!(
        !app.drain_remote_input(),
        "full queue should not report work to run"
    );
    assert_eq!(app.queue.len(), crate::MAX_PROMPT_QUEUE);
    assert!(!app.queue.iter().any(|p| p.starts_with("remote-")));
    assert!(
        app.transcript_text().contains("queue full"),
        "rejected remote prompts should be visible"
    );
}

#[tokio::test]
async fn sessions_reject_path_like_ids_before_callbacks_or_http() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");
    app.session_switcher = Some(Box::new(|_, _| {
        Box::pin(async { panic!("invalid id reached switch callback") })
    }));

    app.handle_sessions_command(&mut agent, "switch ../../escape")
        .await;
    app.handle_sessions_command(&mut agent, "rename ../../escape bad")
        .await;

    assert_eq!(
        app.transcript_text().matches("invalid session id").count(),
        2
    );
}

#[test]
fn sticky_scroll_unpins_on_scroll_up_and_repins_at_bottom() {
    let mut app = test_app("openai", "gpt-4o");
    // Simulate what render() caches for a transcript taller than the viewport.
    app.view_max_scroll = 100;
    app.view_total = 120;
    assert!(app.following, "starts pinned to the bottom");

    // Scrolling up unpins, holds an absolute offset, and snapshots the count.
    app.scroll_up(10);
    assert!(!app.following, "scroll up unpins");
    assert_eq!(app.scroll, 90, "offset = max_scroll - 10");
    assert_eq!(app.total_when_unpinned, 120);

    // Streaming output below must NOT yank a scrolled-up reader back down.
    app.apply(UiEvent::Text {
        text: "a fresh streamed line\n".into(),
    });
    assert!(
        !app.following,
        "new output leaves the scrolled-up reader put"
    );

    // Scrolling back past the bottom re-pins so output follows again.
    app.scroll_down(1000);
    assert!(app.following, "reaching the bottom re-pins");
}

#[test]
fn mouse_wheel_scrolls_and_repins_the_transcript() {
    let mut app = test_app("openai", "gpt-4o");
    app.view_max_scroll = 30;
    app.view_total = 50;

    let wheel = |kind| crossterm::event::MouseEvent {
        kind,
        column: 0,
        row: 0,
        modifiers: KeyModifiers::NONE,
    };
    app.handle_mouse(wheel(crossterm::event::MouseEventKind::ScrollUp));
    assert!(!app.following);
    assert_eq!(app.scroll, 27);

    app.handle_mouse(wheel(crossterm::event::MouseEventKind::ScrollDown));
    assert!(app.following, "wheel-down at the bottom should re-pin");
}

#[tokio::test]
async fn team_command_routes_executors_and_clears_back_to_driver() {
    // Deterministic backend + isolated weights dir: CI has no MLX/CUDA, and
    // an aborted provisioning task must not litter the repo's .hi/models.
    // SAFETY: nextest isolates each test in its own process.
    unsafe { std::env::set_var("HI_LOCAL_BACKEND", "mlx") };
    unsafe {
        std::env::set_var(
            "HI_MLX_MODELS_DIR",
            std::env::temp_dir().join(format!("hi-team-test-{}", std::process::id())),
        )
    };
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");

    // Bare /team renders the role table.
    app.handle_command(&mut agent, hi_agent::Command::Team(String::new()))
        .await;
    let text = app.transcript_text();
    assert!(text.contains("driver"), "role table shown: {text}");
    assert!(text.contains("delegate"), "role table shown: {text}");

    // Route the executors: delegate to a local endpoint, explore by model.
    app.handle_command(
        &mut agent,
        hi_agent::Command::Team("delegate qwen3-coder http://127.0.0.1:18080/v1".into()),
    )
    .await;
    assert!(
        app.transcript_text()
            .contains("delegate → qwen3-coder @ http://127.0.0.1:18080/v1"),
        "{}",
        app.transcript_text()
    );
    let delegate = agent
        .team_roles()
        .into_iter()
        .find(|r| r.role == "delegate")
        .unwrap();
    assert!(!delegate.inherited);
    assert_eq!(delegate.model, "qwen3-coder");

    // `local` is fully automated: it resolves a supported model for this
    // hardware and starts the download/server setup in the background.
    app.handle_command(&mut agent, hi_agent::Command::Team("explore local".into()))
        .await;
    assert!(
        app.transcript_text().contains("locally for explore"),
        "{}",
        app.transcript_text()
    );
    if let Some(pending) = app.pending_team_provision.take() {
        pending.task.abort();
    }

    // `off` returns the role to the driver.
    app.handle_command(&mut agent, hi_agent::Command::Team("delegate off".into()))
        .await;
    assert!(
        agent
            .team_roles()
            .into_iter()
            .find(|r| r.role == "delegate")
            .unwrap()
            .inherited,
        "delegate returned to the driver route"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn team_route_change_cannot_be_overwritten_by_pending_local_setup() {
    // SAFETY: nextest isolates each test in its own process.
    unsafe { std::env::set_var("HI_LOCAL_BACKEND", "mlx") };
    unsafe {
        std::env::set_var(
            "HI_MLX_MODELS_DIR",
            std::env::temp_dir().join(format!("hi-team-cancel-{}", std::process::id())),
        )
    };
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");

    app.handle_command(
        &mut agent,
        hi_agent::Command::Team("delegate coder-7b".into()),
    )
    .await;
    assert!(app.pending_team_provision.is_some());

    // The user changes the route before the download/server startup finishes.
    // The in-flight result must become stale rather than restoring the local
    // route after this explicit `off` command.
    app.handle_command(&mut agent, hi_agent::Command::Team("delegate off".into()))
        .await;
    assert!(
        agent
            .team_roles()
            .into_iter()
            .find(|role| role.role == "delegate")
            .is_some_and(|role| role.inherited)
    );
    assert!(
        app.pending_team_provision
            .as_ref()
            .is_some_and(|pending| pending.cancelled)
    );

    let pending = app.pending_team_provision.take().expect("pending setup");
    pending.task.abort();
    let _ = pending.task.await;
    assert!(
        agent
            .team_roles()
            .into_iter()
            .find(|role| role.role == "delegate")
            .is_some_and(|role| role.inherited)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn newer_local_team_choice_survives_stale_setup_cleanup() {
    // SAFETY: nextest isolates each test in its own process.
    unsafe { std::env::set_var("HI_LOCAL_BACKEND", "mlx") };
    unsafe {
        std::env::set_var(
            "HI_MLX_MODELS_DIR",
            std::env::temp_dir().join(format!("hi-team-replace-{}", std::process::id())),
        )
    };
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");

    app.handle_command(
        &mut agent,
        hi_agent::Command::Team("delegate coder-7b".into()),
    )
    .await;
    assert!(app.pending_team_provision.is_some());

    // The first task is stale, but the replacement is itself a local choice.
    // It must wait for cleanup rather than being rejected as "one setup at a
    // time" and silently leaving the old route selected.
    app.handle_command(
        &mut agent,
        hi_agent::Command::Team("delegate coder-7b".into()),
    )
    .await;
    assert!(
        app.pending_team_provision
            .as_ref()
            .is_some_and(|pending| pending.cancelled)
    );
    assert_eq!(
        app.queued_team_assignments
            .iter()
            .map(|(role, _)| role.as_str())
            .collect::<Vec<_>>(),
        vec!["delegate"]
    );

    app.pending_team_provision
        .as_ref()
        .expect("stale setup")
        .task
        .abort();
    for _ in 0..20 {
        if app
            .pending_team_provision
            .as_ref()
            .is_some_and(|pending| pending.task.is_finished())
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    app.poll_pending_team_provision(&mut agent).await;
    assert!(
        app.pending_team_provision
            .as_ref()
            .is_some_and(|pending| pending.role == "delegate" && !pending.cancelled),
        "the replacement setup should start after stale cleanup"
    );

    if let Some(pending) = app.pending_team_provision.take() {
        pending.task.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn team_supported_model_provisions_in_background_and_wires_on_success() {
    // Deterministic backend + isolated weights dir: CI has no MLX/CUDA, and
    // an aborted provisioning task must not litter the repo's .hi/models.
    // SAFETY: nextest isolates each test in its own process.
    unsafe { std::env::set_var("HI_LOCAL_BACKEND", "mlx") };
    unsafe {
        std::env::set_var(
            "HI_MLX_MODELS_DIR",
            std::env::temp_dir().join(format!("hi-team-test-{}", std::process::id())),
        )
    };
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");

    // A supported name starts a background setup instead of demanding a URL.
    app.handle_command(
        &mut agent,
        hi_agent::Command::Team("delegate coder-7b".into()),
    )
    .await;
    assert!(
        app.transcript_text()
            .contains("setting up coder-7b locally"),
        "{}",
        app.transcript_text()
    );
    assert!(app.pending_team_provision.is_some(), "background task runs");
    assert!(
        agent
            .team_roles()
            .into_iter()
            .find(|r| r.role == "delegate")
            .unwrap()
            .inherited,
        "role stays on the driver until the setup lands"
    );

    // Second request while one is in flight: calm, no second task.
    app.handle_command(
        &mut agent,
        hi_agent::Command::Team("explore coder-7b".into()),
    )
    .await;
    assert!(
        app.transcript_text().contains("one local setup at a time"),
        "{}",
        app.transcript_text()
    );

    // Simulate the provisioning outcome landing (the real task would need a
    // model download; the apply path is what must be correct).
    if let Some(pending) = app.pending_team_provision.take() {
        pending.task.abort();
    }
    let test_process_id = hi_tools::spawn_local_server(
        std::path::Path::new("/bin/sh"),
        &["-c".into(), "sleep 60".into()],
    )
    .expect("test local server process");
    app.apply_team_provision_result(
        &mut agent,
        "delegate",
        "coder-7b",
        Ok((
            "http://127.0.0.1:18080/v1".into(),
            "Qwen2.5-Coder-7B-Instruct-4bit".into(),
            test_process_id.clone(),
        )),
    );
    let delegate = agent
        .team_roles()
        .into_iter()
        .find(|r| r.role == "delegate")
        .unwrap();
    assert_eq!(delegate.model, "Qwen2.5-Coder-7B-Instruct-4bit");
    assert_eq!(delegate.route, "http://127.0.0.1:18080/v1");
    assert!(
        app.transcript_text()
            .contains("✓ delegate → Qwen2.5-Coder-7B-Instruct-4bit @ local"),
        "{}",
        app.transcript_text()
    );

    // The registered server is now reused instantly for another role.
    app.handle_command(
        &mut agent,
        hi_agent::Command::Team("explore coder-7b".into()),
    )
    .await;
    assert!(
        app.transcript_text().contains("reusing the running server"),
        "{}",
        app.transcript_text()
    );
    assert!(app.pending_team_provision.is_none(), "no new task needed");

    hi_tools::stop_local_server(&test_process_id);
    assert!(
        !hi_tools::local_server_is_running(&test_process_id),
        "a stopped child must not be reused"
    );
    assert!(
        agent
            .running_local_model_server("Qwen2.5-Coder-7B-Instruct-4bit")
            .is_none(),
        "a stale team route must not be offered as reusable"
    );

    // Failures stay calm: role unchanged, no raw error dump.
    app.apply_team_provision_result(
        &mut agent,
        "explore",
        "coder-32b",
        Err(anyhow::anyhow!("no local-inference backend detected")),
    );
    assert!(
        app.transcript_text()
            .contains("couldn't set up coder-32b locally"),
        "{}",
        app.transcript_text()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bare_team_opens_role_menu_and_routes_to_model_picker() {
    // SAFETY: nextest isolates each test in its own process.
    unsafe { std::env::set_var("HI_LOCAL_BACKEND", "mlx") };
    unsafe {
        std::env::set_var(
            "HI_MLX_MODELS_DIR",
            std::env::temp_dir().join(format!("hi-team-menu-{}", std::process::id())),
        )
    };
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");

    // Bare `/team` opens the ROLE menu, auto-setup row first.
    app.handle_command(&mut agent, hi_agent::Command::Team(String::new()))
        .await;
    assert!(app.team_role_menu, "role menu flag set");
    let rows = app.picker.as_ref().expect("role menu opens").all.clone();
    assert!(rows[0].starts_with("auto-setup"), "{rows:?}");
    assert!(rows.iter().any(|row| row.starts_with("editor")), "{rows:?}");

    // Enter on the delegate row swaps to that role's MODEL picker.
    let delegate_row = rows
        .iter()
        .position(|row| row.starts_with("delegate"))
        .unwrap();
    if let Some(picker) = app.picker.as_mut() {
        picker.selected = picker
            .matches
            .iter()
            .position(|&i| i == delegate_row)
            .unwrap();
    }
    app.pick_model(&mut agent);
    assert!(!app.team_role_menu, "role menu consumed");
    assert_eq!(app.team_picker_role.as_deref(), Some("delegate"));
    assert!(
        app.picker
            .as_ref()
            .is_some_and(|p| p.all.iter().any(|row| row.starts_with("laguna-s"))),
        "model picker opened for the role"
    );

    // Esc must clear ALL team routing state, or the next /model pick would
    // assign a role instead of switching the driver.
    app.close_picker();
    assert!(app.picker.is_none() && app.team_picker_role.is_none() && !app.team_role_menu);

    // `/team auto` announces the plan, starts delegate provisioning, and
    // queues editor+explore behind it.
    app.handle_command(&mut agent, hi_agent::Command::Team("auto".into()))
        .await;
    assert!(
        app.transcript_text().contains("auto-setup: delegate →"),
        "{}",
        app.transcript_text()
    );
    assert!(
        app.pending_team_provision.is_some(),
        "delegate provisioning starts"
    );
    let queued: Vec<&str> = app
        .queued_team_assignments
        .iter()
        .map(|(role, _)| role.as_str())
        .collect();
    assert_eq!(queued, vec!["editor", "explore"]);
    assert!(app.auto_setup_skeptic);

    // Explicitly disabling the skeptic while the chain is in flight must stop
    // the queued auto-enable from undoing that choice later.
    app.handle_command(
        &mut agent,
        hi_agent::Command::Config("skeptic-local off".into()),
    )
    .await;
    assert!(app.queued_team_assignments.is_empty());
    assert!(!app.auto_setup_skeptic);

    // A queued role must remain on the user's explicit route even though a
    // different role's download is still in flight.
    app.handle_command(&mut agent, hi_agent::Command::Team("explore off".into()))
        .await;
    assert!(app.queued_team_assignments.is_empty());
    assert!(!app.auto_setup_skeptic);
    assert!(
        agent
            .team_roles()
            .into_iter()
            .find(|role| role.role == "explore")
            .is_some_and(|role| role.inherited)
    );
    if let Some(pending) = app.pending_team_provision.take() {
        pending.task.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bare_team_role_opens_picker_and_selection_starts_setup() {
    // SAFETY: nextest isolates each test in its own process.
    unsafe { std::env::set_var("HI_LOCAL_BACKEND", "mlx") };
    unsafe {
        std::env::set_var(
            "HI_MLX_MODELS_DIR",
            std::env::temp_dir().join(format!("hi-team-picker-{}", std::process::id())),
        )
    };
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");

    // `/team delegate` with no model opens the catalog picker.
    app.handle_command(&mut agent, hi_agent::Command::Team("delegate".into()))
        .await;
    assert!(app.picker.is_some(), "picker opens");
    assert_eq!(app.team_picker_role.as_deref(), Some("delegate"));
    let rows = app.picker.as_ref().unwrap().all.clone();
    assert!(
        rows.iter().any(|row| row.starts_with("laguna-s")),
        "laguna listed: {rows:?}"
    );
    assert!(
        rows.iter().any(|row| row.starts_with("qwen3.6-35b")),
        "qwen3.6 listed: {rows:?}"
    );
    assert!(
        rows.iter()
            .any(|row| row.starts_with("glm-5.2") && row.contains("too big")),
        "oversized entries are annotated honestly: {rows:?}"
    );

    // Selecting a row routes to team setup, not a driver-model switch.
    let coder_row_index = app
        .picker
        .as_ref()
        .unwrap()
        .all
        .iter()
        .position(|row| row.starts_with("coder-7b"))
        .expect("coder-7b row");
    if let Some(picker) = app.picker.as_mut() {
        picker.selected = picker
            .matches
            .iter()
            .position(|&i| i == coder_row_index)
            .unwrap_or(0);
    }
    app.pick_model(&mut agent);
    assert!(app.picker.is_none(), "picker closes");
    assert!(app.team_picker_role.is_none());
    assert_eq!(
        "gpt-4o", app.model,
        "the driver model must NOT change from a team pick"
    );
    assert!(
        app.transcript_text()
            .contains("setting up coder-7b locally"),
        "{}",
        app.transcript_text()
    );
    if let Some(pending) = app.pending_team_provision.take() {
        pending.task.abort();
    }
}

#[tokio::test]
async fn config_command_sets_disables_and_restores_automatic_step_limit() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");
    assert_eq!(
        agent.max_steps_setting(),
        "32",
        "finite automatic cap by default"
    );

    app.handle_command(&mut agent, hi_agent::Command::Config("steps 350".into()))
        .await;
    assert_eq!(agent.max_steps_setting(), "350");

    app.handle_command(&mut agent, hi_agent::Command::Config("steps off".into()))
        .await;
    assert_eq!(agent.max_steps_setting(), "off");

    app.handle_command(&mut agent, hi_agent::Command::Config("steps 350".into()))
        .await;
    app.handle_command(&mut agent, hi_agent::Command::Config("steps auto".into()))
        .await;
    assert_eq!(agent.max_steps_setting(), "32", "auto restores the default");
    assert!(app.transcript_text().contains("step limit → 32 (automatic"));
}

#[test]
fn transcript_is_capped_while_following_but_not_while_scrolled_up() {
    let mut app = test_app("openai", "gpt-4o");
    // Following stays bounded and keeps the newest lines.
    for i in 0..(MAX_TRANSCRIPT_LINES + 5_000) {
        app.push(Line::raw(format!("l{i}")));
    }
    assert_eq!(
        app.transcript.len(),
        MAX_TRANSCRIPT_LINES,
        "bounded while following"
    );
    assert_eq!(
        app.transcript.last().unwrap().text(),
        format!("l{}", MAX_TRANSCRIPT_LINES + 5_000 - 1),
        "newest line kept"
    );

    // Scrolled-up pushes are not trimmed because that would shift reader offsets.
    app.view_max_scroll = 50;
    app.view_total = 60;
    app.scroll_up(5);
    assert!(!app.following, "scrolled up");
    let before = app.transcript.len();
    for i in 0..1_000 {
        app.push(Line::raw(format!("m{i}")));
    }
    assert_eq!(
        app.transcript.len(),
        before + 1_000,
        "grows while scrolled up, no trim"
    );
}

#[test]
fn scrolling_moves_the_viewport_through_render_and_repins() {
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..100 {
        app.push(Line::raw(format!("line {i:03}")));
    }
    let mut term = Terminal::new(TestBackend::new(40, 12)).unwrap();
    // Following: the bottom is visible, the top is not.
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("line 099"),
        "bottom visible when following:\n{screen}"
    );
    assert!(
        !screen.contains("line 000"),
        "top hidden when following:\n{screen}"
    );

    // Scroll up: earlier lines appear, the bottom leaves the viewport.
    app.scroll_up(40);
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(!app.following, "scroll up unpins");
    assert!(
        !screen.contains("line 099"),
        "bottom gone after scroll up:\n{screen}"
    );
    assert!(
        screen.contains("line 0"),
        "older lines now visible:\n{screen}"
    );

    // Scroll back down past the end: re-pins and shows the bottom again.
    app.scroll_down(1000);
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(app.following, "re-pinned at the bottom");
    assert!(
        screen.contains("line 099"),
        "bottom visible again:\n{screen}"
    );
}

#[test]
fn following_shows_last_line_when_word_wrapping_creates_extra_rows() {
    // Regression: `wrapped_height` counted characters (ceil(len/width)) but
    // ratatui's `WordWrapper` wraps at word boundaries — a word that doesn't
    // fit the remaining space moves to the next line, and a word wider than
    // the line is broken across rows. That makes the real wrapped height
    // LARGER than the char-based estimate, so `max_scroll` was too small
    // and the bottom of a long message was clipped off-screen.
    //
    // Each line below has a 45-char word at width 38: ratatui wraps it to
    // 3 rows, but the old char-based estimate said 2. With 20 such lines
    // the ~20-row undercount pushed the last line entirely off-screen.
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..20 {
        app.push(Line::raw(format!(
            "word{i:02} supercalifragilisticexpialidocious_extras"
        )));
    }
    app.push(Line::raw("LAST_LINE_MARKER_42"));

    let mut term = Terminal::new(TestBackend::new(40, 12)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("LAST_LINE_MARKER_42"),
        "last line must be visible when following (word-wrap clip bug):\n{screen}"
    );
}

#[test]
fn following_shows_last_line_with_realistic_prose_word_wrapping() {
    // A second regression check with normal prose (no artificially long
    // words). At a narrow width, word-boundary wrapping still produces more
    // rows than char-based `ceil(len/width)` because words that don't fit
    // the remaining space leave the current line short. This is the case
    // that clipped the end of a long assistant message in practice.
    let mut app = test_app("openai", "gpt-4o");
    // 30 lines of prose, each ~70 chars. At width 36 (inner of a 38-wide
    // terminal), char-based says ceil(70/36) = 2 rows per line, but
    // word-wrap often produces 3 because words straddle the boundary.
    for i in 0..30 {
        app.push(Line::raw(format!(
            "The quick brown fox jumps over the lazy dog and then runs {i:02}"
        )));
    }
    app.push(Line::raw("FINAL_ANSWER_99"));

    let mut term = Terminal::new(TestBackend::new(38, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("FINAL_ANSWER_99"),
        "last line must be visible with prose word-wrapping:\n{screen}"
    );
}

#[test]
fn working_line_tracks_model_phase_not_tool_ids() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    // Model phase: reasoning then text stream distinctly.
    app.apply(UiEvent::Reasoning { text: "hmm".into() });
    assert!(
        app.activity_line().starts_with("thinking…"),
        "{}",
        app.activity_line()
    );
    app.apply(UiEvent::Text {
        text: "here".into(),
    });
    assert!(
        app.activity_line().starts_with("responding…"),
        "{}",
        app.activity_line()
    );
    // A tool in flight keeps the quiet Working wave — command identity is
    // a transcript `Run` row, not this chrome line.
    app.turn_rounds = 2;
    app.turn_tool_calls = 1;
    app.apply(UiEvent::ToolStarted {
        name: "bash".into(),
        arguments: "{\"command\":\"cargo test\"}".into(),
    });
    let line = app.activity_line();
    assert!(line.starts_with("Working"), "{line}");
    assert!(
        !line.contains("bash") && !line.contains("round") && !line.contains("call"),
        "{line}"
    );
    app.apply(UiEvent::ToolCall {
        name: "bash_output".into(),
        arguments: "{\"id\":\"cargo-test_2\"}".into(),
    });
    let line = app.activity_line();
    assert!(line.starts_with("Working"), "{line}");
    assert!(
        !line.contains("bash_output") && !line.contains("cargo-test_2"),
        "{line}"
    );
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines.iter().any(|l| l.contains("Run cargo-test_2")),
        "live Run row: {lines:?}"
    );
    app.apply(UiEvent::ToolResult {
        name: "bash_output".into(),
        result: "[cargo-test_2: still running — no new output]".into(),
    });
    assert!(
        app.activity_line().starts_with("Working"),
        "{}",
        app.activity_line()
    );
}

#[test]
fn working_wave_sweeps_one_lit_letter_at_a_time() {
    // "Working" is 7 letters; the lit index sweeps 0..6 then 6..0.
    // At each tick exactly one letter is white/bold and the rest are gray.
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    let n = "Working".chars().count();
    let cycle = 2 * (n - 1);
    for tick in 0..cycle {
        app.spinner = tick;
        let spans = app.working_spans();
        assert_eq!(spans.len(), n, "one span per letter at tick {tick}");
        let lit_count = spans
            .iter()
            .filter(|s| s.style.fg == Some(crate::theme::theme().accent_running))
            .count();
        assert_eq!(lit_count, 1, "exactly one lit letter at tick {tick}");
        // The lit index matches the forward/back sweep.
        let expected_lit = if tick < n { tick } else { cycle - tick };
        assert_eq!(
            spans[expected_lit].style.fg,
            Some(crate::theme::theme().accent_running),
            "lit index {expected_lit} at tick {tick}"
        );
    }
}

#[test]
fn renders_tool_call_diff_and_spinner() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"src/cli.rs\",\"old_string\":\"a\",\"new_string\":\"b\"}".into(),
    });
    // ANSI-colored diff line (from the edit tool) must render as text.
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "\u{1b}[32m+ pub json: bool\u{1b}[0m".into(),
    });
    app.apply(UiEvent::TurnEnd {
        summary: "[1234 in · 56 out · 1290 total]".into(),
    });
    app.working = true;
    app.spinner = 2;

    let mut term = Terminal::new(TestBackend::new(56, 13)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    // The header reads as "Edit <basename> +N/-M", not a raw JSON dump.
    assert!(
        screen.contains("Edit cli.rs"),
        "readable edit header: {screen}"
    );
    assert!(screen.contains("+1"), "collapsed edit shows +N: {screen}");
    assert!(
        !screen.contains("old_string"),
        "header must not dump JSON args"
    );
    let copied = app
        .transcript
        .iter()
        .map(TranscriptEntry::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        copied.contains("pub json: bool"),
        "copy/export keeps the diff body"
    );
    assert!(
        screen.contains(SPINNER[2]) && screen.contains("0s"),
        "prompt bar shows the spinner + an elapsed timer while working: {screen}"
    );
    assert!(
        screen.contains("[stop]"),
        "prompt bar shows [stop] while working: {screen}"
    );
}

#[test]
fn colorizes_plain_diff_tool_output() {
    let mut app = test_app("openai", "gpt-4o");
    let diff = "--- a/x.rs\n+++ b/x.rs\n@@ -1,2 +1,2 @@\n-old\n+new\n ctx\n";
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: diff.into(),
    });
    // The content span (after the "  " indent) carries the diff color.
    let colored: Vec<(String, Option<Color>)> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, true, crate::Density::Comfortable))
        .map(|l| {
            let text: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
            (text, l.spans.last().map(|s| s.style.fg).unwrap_or(None))
        })
        .collect();
    assert!(
        colored
            .iter()
            .any(|(t, fg)| t.contains("+new") && *fg == Some(crate::theme::theme().diff_add)),
        "added line is green: {colored:?}"
    );
    assert!(
        colored
            .iter()
            .any(|(t, fg)| t.contains("-old") && *fg == Some(crate::theme::theme().diff_del)),
        "removed line is red"
    );
    assert!(
        colored
            .iter()
            .any(|(t, fg)| t.contains("@@") && *fg == Some(crate::theme::theme().diff_hunk)),
        "hunk header is cyan"
    );
}

#[test]
fn non_diff_tool_output_is_not_colorized() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "- item one\n- item two\n".into(),
    });
    let any_red = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, true, crate::Density::Comfortable))
        .any(|l| l.spans.last().map(|s| s.style.fg) == Some(Some(Color::Red)));
    assert!(!any_red, "a plain list must not be colorized as a diff");
}

#[test]
fn usage_event_keeps_tokens_out_of_compact_working_line() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.apply(UiEvent::Usage {
        prompt: 12,
        generated: 340,
        ctx_used: 64_000,
        ctx_window: Some(128_000),
        estimated: false,
    });
    assert_eq!(app.usage, (12, 340));
    assert_eq!(app.context_pct(), Some(50));

    let mut term = Terminal::new(TestBackend::new(72, 8)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains(SPINNER[0]), "spinner shown: {screen}");
    assert!(
        !screen.contains("prompt↑"),
        "no duplicate prompt tokens: {screen}"
    );
    assert!(
        !screen.contains("gen↓"),
        "no duplicate output tokens: {screen}"
    );
    assert!(screen.contains("64k / 128k"), "live context fill: {screen}");
}

#[test]
fn session_cost_chip_hidden_without_price() {
    let mut app = test_app("openai", "gpt-4o");
    app.session_totals = hi_ai::Usage {
        input_tokens: 1_000_000,
        output_tokens: 0,
        ..hi_ai::Usage::default()
    };
    assert!(app.session_cost_chip().is_none());
    app.usage_pricing = Some((2.0, 4.0));
    assert_eq!(app.session_cost_chip().as_deref(), Some("$2.00"));

    let mut term = Terminal::new(TestBackend::new(80, 8)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("$2.00"), "cost chip: {screen}");

    app.usage_pricing = None;
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        !screen.contains("$2.00"),
        "chip hidden without price: {screen}"
    );
}

#[tokio::test]
async fn apply_metadata_marks_new_ids_and_wires_pricing() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "old");
    app.model_ids = vec!["old".into()];
    let served = vec![
        hi_ai::ServedModel {
            id: "old".into(),
            context_window: Some(128_000),
            max_output_tokens: None,
            price: Some((1.0, 2.0)),
            provider_label: None,
            status: None,
            available: true,
            availability_reason: None,
            capabilities: Vec::new(),
        },
        hi_ai::ServedModel {
            id: "fresh".into(),
            context_window: None,
            max_output_tokens: None,
            price: None,
            provider_label: None,
            status: None,
            available: true,
            availability_reason: None,
            capabilities: Vec::new(),
        },
    ];
    apply_metadata(&mut app, &mut agent, &Ok(served), "test-cache-key");
    assert!(app.new_model_ids.contains("fresh"));
    assert!(!app.new_model_ids.contains("old"));
    assert_eq!(agent.usage_pricing(), Some((1.0, 2.0)));
    let mut picker = app.model_picker(vec!["old".into(), "fresh".into()], "old");
    picker.mark_new(&app.new_model_ids);
    assert!(picker.meta.get("fresh").is_some_and(|m| m.is_new));
}

#[test]
fn completed_turn_latency_is_visible_in_idle_status() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.started = Some(std::time::Instant::now() - std::time::Duration::from_secs(3));
    app.last_turn_state = TurnState::Done("done".into());
    app.set_working(false);

    let mut term = Terminal::new(TestBackend::new(100, 8)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Worked for") && !screen.contains("last: done"),
        "idle latency is a Worked for marker, not a done row: {screen}"
    );
}

#[test]
fn rate_limit_event_updates_working_line() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.apply(UiEvent::RateLimits {
        rate_limits: Some(hi_ai::RateLimitState {
            requests_min: hi_ai::RateLimitBucket {
                limit: 60,
                remaining: 58,
                reset_seconds: 12,
            },
            tokens_min: hi_ai::RateLimitBucket {
                limit: 100_000,
                remaining: 88_000,
                reset_seconds: 42,
            },
            ..Default::default()
        }),
    });

    let mut term = Terminal::new(TestBackend::new(100, 8)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("limits req 58/60"),
        "request bucket: {screen}"
    );
    assert!(screen.contains("tok 88k/100k"), "token bucket: {screen}");
}

#[test]
fn renders_queued_commands_while_working() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.queue.push_back("run the tests".into());
    app.queue.push_back("then commit".into());
    app.input.set("typing a third");

    let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    assert!(screen.contains(SPINNER[0]), "spinner shown while working");
    assert!(
        screen.contains("run the tests"),
        "first queued command shown"
    );
    assert!(
        screen.contains("then commit"),
        "second queued command shown"
    );
    assert!(
        screen.contains("typing a third"),
        "input stays editable while working"
    );
}

#[test]
fn renders_pinned_plan_checklist() {
    use hi_agent::PlanStep;
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Plan {
        steps: vec![
            PlanStep {
                title: "find leak".into(),
                status: PlanStatus::Done,
            },
            PlanStep {
                title: "fix walkers".into(),
                status: PlanStatus::Active,
            },
            PlanStep {
                title: "add tests".into(),
                status: PlanStatus::Pending,
            },
        ],
    });

    let mut term = Terminal::new(TestBackend::new(60, 20)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    assert!(
        screen.contains("plan · 1/3"),
        "plan header w/ progress:\n{screen}"
    );
    assert!(screen.contains("find leak"), "step titles shown:\n{screen}");
    assert!(screen.contains("fix walkers"));
    assert!(screen.contains("add tests"));
    assert!(screen.contains('✓'), "done glyph:\n{screen}");
    assert!(screen.contains('▸'), "active glyph:\n{screen}");

    // A later update replaces the plan in place — progress advances and the
    // checklist isn't duplicated into the transcript.
    app.apply(UiEvent::Plan {
        steps: vec![
            PlanStep {
                title: "find leak".into(),
                status: PlanStatus::Done,
            },
            PlanStep {
                title: "fix walkers".into(),
                status: PlanStatus::Done,
            },
            PlanStep {
                title: "add tests".into(),
                status: PlanStatus::Active,
            },
        ],
    });
    term.draw(|f| app.render(f)).unwrap();
    let screen2 = dump(&term);
    assert!(
        screen2.contains("plan · 2/3"),
        "progress advanced:\n{screen2}"
    );
    assert!(
        app.transcript.is_empty(),
        "plan must not echo into the transcript"
    );

    app.apply(UiEvent::Plan { steps: Vec::new() });
    term.draw(|f| app.render(f)).unwrap();
    let screen3 = dump(&term);
    assert!(
        !screen3.contains("plan ·"),
        "empty update clears box:\n{screen3}"
    );
}

#[test]
fn long_plan_does_not_break_input_box_border() {
    // Regression: when the plan + status + input is taller than the screen,
    // the input box height used to exceed the terminal height. ratatui's
    // Layout clamps the fixed-Length rect, so the Paragraph content spilled
    // past the bottom border — the `╰` landed mid-content and later steps
    // rendered outside the box. The box must stay closed and fit on screen.
    use hi_agent::PlanStep;
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Plan {
        steps: (0..8)
            .map(|i| PlanStep {
                title: format!("step {i} with a fairly long title to be realistic"),
                status: if i < 3 {
                    PlanStatus::Done
                } else if i == 3 {
                    PlanStatus::Active
                } else {
                    PlanStatus::Pending
                },
            })
            .collect(),
    });
    app.working = true;
    // Tiny height: the full plan (9 lines) + status + input + borders can't fit.
    let mut term = Terminal::new(TestBackend::new(80, 12)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    // The bottom border must close on its own row, not overlap a plan step.
    let bottom_rows: Vec<&str> = screen.lines().filter(|l| l.contains('╰')).collect();
    assert_eq!(
        bottom_rows.len(),
        1,
        "exactly one bottom-left corner:\n{screen}"
    );
    // The corner must be the first non-space glyph on its row (a closed
    // border), not sitting on top of a `✓`/`▸`/`☐` step glyph.
    let row = bottom_rows[0];
    assert!(
        row.trim_start().starts_with('╰'),
        "bottom border row must start with `╰`, got: {row:?}\n{screen}"
    );
    // The plan is truncated to fit, with a "… +N more" line, rather than
    // overflowing.
    assert!(
        screen.contains("… +") && screen.contains("more"),
        "plan truncated to fit:\n{screen}"
    );
    // The box never exceeds the terminal height.
    assert!(
        screen.lines().filter(|l| !l.trim().is_empty()).count() <= 12,
        "box fits on screen:\n{screen}"
    );

    // A taller terminal shows the whole plan with no truncation.
    let mut term2 = Terminal::new(TestBackend::new(175, 18)).unwrap();
    term2.draw(|f| app.render(f)).unwrap();
    let screen2 = dump(&term2);
    assert!(
        screen2.contains("step 7 with a fairly long title to be realistic"),
        "full plan shown when it fits:\n{screen2}"
    );
    assert!(!screen2.contains("… +"), "no truncation when it fits");

    // Extreme case: a plan so large the box would fill the whole screen.
    // The transcript must still get its Min(1) row and the border must stay
    // closed — the cap reserves a row for the transcript so Layout never
    // clamps the box rect.
    let mut app2 = test_app("openai", "gpt-4o");
    app2.apply(UiEvent::Plan {
        steps: (0..20)
            .map(|i| PlanStep {
                title: format!("step {i}"),
                status: PlanStatus::Pending,
            })
            .collect(),
    });
    app2.working = true;
    let mut term3 = Terminal::new(TestBackend::new(60, 10)).unwrap();
    term3.draw(|f| app2.render(f)).unwrap();
    let screen3 = dump(&term3);
    let bottom3: Vec<&str> = screen3.lines().filter(|l| l.contains('╰')).collect();
    assert_eq!(bottom3.len(), 1, "one bottom corner:\n{screen3}");
    assert!(
        bottom3[0].trim_start().starts_with('╰'),
        "border closed, not overlapping content:\n{screen3}"
    );
    // Status bar (cwd) and prompt (model in the bottom divider) both survive.
    assert!(
        screen3.contains("gpt-4o"),
        "transcript keeps its row:\n{screen3}"
    );

    // Degenerate tiny terminal: must not panic, and the box border stays closed.
    let mut term4 = Terminal::new(TestBackend::new(60, 3)).unwrap();
    term4.draw(|f| app2.render(f)).unwrap();
    let screen4 = dump(&term4);
    let bottom4: Vec<&str> = screen4.lines().filter(|l| l.contains('╰')).collect();
    assert_eq!(
        bottom4.len(),
        1,
        "one bottom corner on tiny term:\n{screen4}"
    );
}

#[test]
fn startup_notice_does_not_clip_input_line() {
    // On first load, a startup notice (e.g. "model metadata not loaded: …")
    // is pinned above the status line. The box height must
    // account for it, or the input line gets clipped and the cursor lands
    // on the wrong row.
    let mut app = test_app("openai", "gpt-4o");
    app.startup_notice = Some("model metadata not loaded: connection refused".into());
    let mut term = Terminal::new(TestBackend::new(70, 10)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("model metadata not loaded"),
        "notice shown:\n{screen}"
    );
    // The input prompt must still be visible inside the box (not clipped).
    assert!(screen.contains('❯'), "input prompt visible:\n{screen}");
    // The input box's bottom border closes cleanly (the transcript block
    // also has a `╰`, so check the last one — the input box is at the bottom).
    let bottom: Vec<&str> = screen.lines().filter(|l| l.contains('╰')).collect();
    let input_box_border = bottom.last().expect("input box bottom border");
    assert!(
        input_box_border.trim_start().starts_with('╰'),
        "input box border closed:\n{screen}"
    );
    // The notice, status, and prompt all render inside the input box —
    // i.e. above the input box's bottom border row (the last `╰…─` row).
    let rows: Vec<&str> = screen.lines().collect();
    let border_row_idx = rows
        .iter()
        .rposition(|l| l.trim_start().starts_with('╰') && l.contains('─'))
        .unwrap();
    let above_border: String = rows[..border_row_idx].join("\n");
    assert!(
        above_border.contains("model metadata not loaded") && above_border.contains('❯'),
        "notice + prompt above the border:\n{screen}"
    );
}

#[test]
fn quit_notice_renders_and_does_not_clip_input() {
    // After the first Ctrl-C (idle, empty input), a "Press Ctrl-C again to
    // exit" notice is pinned above the status line. The box height must
    // account for it or the input line clips and the cursor lands wrong.
    let mut app = test_app("openai", "gpt-4o");
    app.quit_notice = Some(Instant::now() + Duration::from_millis(1800));
    let mut term = Terminal::new(TestBackend::new(70, 10)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Press Ctrl-C again to exit"),
        "quit notice shown:\n{screen}"
    );
    assert!(screen.contains('❯'), "input prompt visible:\n{screen}");
    // The input box bottom border closes cleanly (not overlapping content).
    let bottom: Vec<&str> = screen.lines().filter(|l| l.contains('╰')).collect();
    let input_box_border = bottom.last().expect("input box bottom border");
    assert!(
        input_box_border.trim_start().starts_with('╰'),
        "input box border closed:\n{screen}"
    );
}

#[test]
fn long_single_line_input_wraps_and_cursor_tracks_it() {
    // The bug: a long single-line prompt didn't wrap — it ran off the right
    // edge and newly typed text was invisible. The input must soft-wrap to the
    // box width, and the cursor must land on the wrapped row/column where the
    // caret actually is.
    let mut app = test_app("openai", "gpt-4o");
    // 40 chars of text on a 28-wide inner area. prefix = 2, so wrap_w = 26
    // → two display lines: first 26 chars, then the remaining 14.
    let long = "abcdefghijklmnopqrstuvwxyz0123456789abcd";
    app.input.set(long);
    // Cursor at the very end (typing position).
    let (lines, cursor_row, cursor_col) = app.input_view(28);
    // First display line holds the first 26 chars; second holds the rest.
    assert!(
        lines.iter().any(|l| l.to_string().contains("abcdefghij")),
        "first wrapped chunk visible"
    );
    assert!(
        lines.iter().any(|l| l.to_string().contains("abcd")),
        "tail wrapped onto a second line: {:?}",
        lines
    );
    // Cursor is on the second wrapped row (index 1), past its last char.
    assert_eq!(cursor_row, 1, "cursor on wrapped row 1");
    // Second chunk is 14 chars + 2-col prefix → cursor at col 16.
    assert_eq!(cursor_col, 16, "cursor col tracks wrap");
}

#[test]
fn long_input_cursor_in_first_wrapped_chunk_stays_on_row_zero() {
    let mut app = test_app("openai", "gpt-4o");
    let long = "abcdefghijklmnopqrstuvwxyz0123456789abcd";
    app.input.set(long);
    // Move cursor to column 5 (within the first wrapped chunk).
    app.input.cursor = 5;
    let (_lines, cursor_row, cursor_col) = app.input_view(28);
    assert_eq!(cursor_row, 0, "cursor on first wrapped row");
    assert_eq!(cursor_col, 2 + 5, "cursor col = prefix + 5");
}

#[test]
fn wide_input_glyphs_wrap_and_position_the_cursor_by_display_columns() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("aa界bb");

    // input_view(6) leaves four columns for content after the `❯ ` prefix;
    // `aa界` occupies exactly four display columns and `bb` wraps below it.
    let (lines, cursor_row, cursor_col) = app.input_view(6);
    let rendered: Vec<String> = lines.iter().map(ToString::to_string).collect();
    assert_eq!(rendered, vec!["❯ aa界", "  bb"]);
    assert_eq!((cursor_row, cursor_col), (1, 4));
}

#[test]
fn empty_input_uses_grok_prompt_without_placeholder() {
    let app = test_app("openai", "gpt-4o");
    let (lines, cursor_row, cursor_col) = app.input_view(80);

    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].to_string(), "❯ ");
    assert_eq!((cursor_row, cursor_col), (0, 2));
}

#[test]
fn empty_input_shows_suggested_prompt_as_ghost_text() {
    let mut app = test_app("openai", "gpt-4o");
    app.suggested_prompt = Some("Run the unit tests".into());
    let (lines, cursor_row, cursor_col) = app.input_view(80);

    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].to_string(), "❯ Run the unit tests");
    assert_eq!((cursor_row, cursor_col), (0, 2));
}

#[test]
fn ghost_shrinks_when_typing_a_matching_prefix() {
    let mut app = test_app("openai", "gpt-4o");
    app.suggested_prompt = Some("Run the unit tests".into());
    app.input.set("Run the");
    let (lines, ..) = app.input_view(80);
    assert_eq!(lines[0].to_string(), "❯ Run the unit tests");
    assert_eq!(app.ghost_suffix(), Some(" unit tests"));
}

#[test]
fn ghost_accept_inserts_remaining_suffix() {
    let mut app = test_app("openai", "gpt-4o");
    app.suggested_prompt = Some("open a PR".into());
    app.input.set("open");
    assert!(app.accept_suggested_prompt());
    assert_eq!(app.input.text(), "open a PR");
    assert!(app.suggested_prompt.is_none());
}

#[test]
fn ghost_esc_on_empty_dismisses_until_next_suggestion() {
    let mut app = test_app("openai", "gpt-4o");
    app.suggested_prompt = Some("Run the unit tests".into());
    app.dismiss_suggested_prompt();
    let (lines, ..) = app.input_view(80);
    assert_eq!(lines[0].to_string(), "❯ ");
    assert!(app.ghost_suffix().is_none());
}

#[test]
fn accept_suggested_prompt_fills_empty_input() {
    let mut app = test_app("openai", "gpt-4o");
    app.suggested_prompt = Some("open a PR".into());
    assert!(app.accept_suggested_prompt());
    assert_eq!(app.input.text(), "open a PR");
    assert!(app.suggested_prompt.is_none());
}

#[test]
fn suggested_prompt_event_ignored_while_typing() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("hello");
    app.apply(UiEvent::SuggestedPrompt {
        text: "should not land".into(),
    });
    assert!(app.suggested_prompt.is_none());
    assert_eq!(app.input.text(), "hello");
}

#[test]
fn suggested_prompt_accepted_while_working_end_of_turn() {
    // Regression: suggest emits inside run_turn before set_working(false).
    // Dropping on `working` made post-turn ghost text dead.
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.apply(UiEvent::SuggestedPrompt {
        text: "Run the unit tests".into(),
    });
    assert_eq!(
        app.suggested_prompt.as_deref(),
        Some("Run the unit tests"),
        "suggestion must stick when applied at end of turn while working"
    );
    app.set_working(false);
    assert_eq!(app.suggested_prompt.as_deref(), Some("Run the unit tests"));
    // Next turn start clears it.
    app.set_working(true);
    assert!(app.suggested_prompt.is_none());
}

#[test]
fn suggested_prompt_skipped_when_queue_nonempty() {
    let mut app = test_app("openai", "gpt-4o");
    assert!(app.try_enqueue_prompt("queued follow-up"));
    app.apply(UiEvent::SuggestedPrompt {
        text: "should not land".into(),
    });
    assert!(app.suggested_prompt.is_none());
}

#[test]
fn multiline_input_uses_aligned_continuation_rows() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("first line\nsecond line");

    let (lines, cursor_row, cursor_col) = app.input_view(80);
    let rendered: Vec<String> = lines.iter().map(ToString::to_string).collect();

    assert_eq!(rendered, vec!["❯ first line", "  second line"]);
    assert_eq!((cursor_row, cursor_col), (1, 13));
}

#[test]
fn keybindings_help_does_not_advertise_idle_escape_or_ctrl_d_quit() {
    let mut app = test_app("openai", "gpt-4o");
    app.show_help = true;
    // The help panel does not scroll — it renders every binding and truncates
    // at the viewport. Size the terminal to the full list so this test checks
    // what the panel says, not what happens to fit; `help_panel_fits_in_*`
    // below is what guards the height itself.
    // One taller than the previous baseline: outer vpad + chrome gaps take
    // four rows that used to belong to the help panel.
    let mut term = Terminal::new(TestBackend::new(80, 56)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    assert!(
        screen.contains("Ctrl-D") && screen.contains("full-screen diff review"),
        "Ctrl-D help should describe diff toggle:\n{screen}"
    );
    assert!(
        screen.contains("/quit") && screen.contains("quit"),
        "explicit quit command should be shown:\n{screen}"
    );
    assert!(
        !screen.contains("Ctrl-D (idle)") && !screen.contains("quit when empty"),
        "help should not advertise the old accidental-exit bindings:\n{screen}"
    );
}

#[test]
fn the_voice_indicator_is_drawn_in_the_input_area() {
    // Proves the wiring, not just the helper: an open microphone has to reach
    // an actual frame, near the prompt where it will be seen.
    let mut app = test_app("openai", "gpt-4o");
    let mut term = Terminal::new(TestBackend::new(80, 20)).unwrap();

    term.draw(|f| app.render(f)).unwrap();
    assert!(
        !dump(&term).contains("transcribing"),
        "nothing voice-related renders while idle"
    );

    let (_tx, rx) = tokio::sync::oneshot::channel();
    app.voice = crate::app::voice::VoiceState::Transcribing { rx };
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("transcribing"),
        "the transcribing status reaches the frame:\n{screen}"
    );
}

#[test]
fn help_panel_still_fits_in_its_documented_height() {
    // The keybindings panel does not scroll: it renders every binding and lets
    // the viewport truncate. So each binding added costs a row off the bottom,
    // and the bottom is where the Sessions section lives.
    //
    // This pins the height the panel currently needs. If you add a binding and
    // this fails, that is the point — either trim some help text, or make the
    // panel scroll, rather than letting `/quit` silently vanish for anyone on a
    // shorter terminal.
    const REQUIRED_ROWS: u16 = 56;
    let mut app = test_app("openai", "gpt-4o");
    app.show_help = true;
    let mut term = Terminal::new(TestBackend::new(80, REQUIRED_ROWS)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("/quit"),
        "the last help entry must be visible at {REQUIRED_ROWS} rows:\n{screen}"
    );
}

#[test]
fn changed_files_line_shows_what_last_turn_touched() {
    // After a turn that changed files, a compact "changed: …" line sits
    // above the input so the user sees what was touched without scrolling.
    let mut app = test_app("openai", "gpt-4o");
    app.last_changed_files = vec!["src/a.rs".into(), "src/b.rs".into()];
    let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("changed: src/a.rs, src/b.rs"),
        "changed-files line: {screen}"
    );
    assert!(
        screen.contains("Ctrl-G for review"),
        "diff toggle hint: {screen}"
    );
}

#[test]
fn ctrl_d_toggles_diff_even_when_input_is_empty() {
    let mut app = test_app("openai", "gpt-4o");
    let ctrl_d = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL);

    assert_eq!(app.edit_key(&ctrl_d), None);
    assert!(
        app.mode.is_review(),
        "Ctrl-D should open full-screen review"
    );
    assert!(app.diff_text.is_some(), "opening should cache diff text");

    assert_eq!(app.edit_key(&ctrl_d), None);
    assert!(
        !app.mode.is_review(),
        "second Ctrl-D should close the review overlay"
    );
}

#[test]
fn ctrl_d_opens_the_full_screen_review() {
    // Ctrl-D is an alias for Ctrl-G: the composer no longer dumps a 20-line
    // git diff. Set diff_text directly to avoid a real git call.
    let mut app = test_app("openai", "gpt-4o");
    app.diff_text = Some("--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-old\n+new\n".into());
    app.mode = crate::mode::UiMode::Review;
    let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Diff review (Ctrl-G)"),
        "review header: {screen}"
    );
    assert!(screen.contains("+new"), "diff content rendered: {screen}");

    app.mode.to_insert();
    term.draw(|f| app.render(f)).unwrap();
    let screen2 = dump(&term);
    assert!(
        !screen2.contains("Diff review (Ctrl-G)"),
        "review closed: {screen2}"
    );
}

#[test]
fn ctrl_question_toggles_the_observability_panel() {
    // The Ctrl-? agent-observability panel renders the last turn's telemetry
    // counters, the per-turn tool-call count, and session/context numbers.
    let mut app = test_app("openai", "gpt-4o");
    app.show_debug = true;
    let mut repair_counts = std::collections::BTreeMap::new();
    repair_counts.insert("review_listing_only".to_string(), 4);
    repair_counts.insert("review_no_evidence".to_string(), 1);
    app.last_telemetry = Some(hi_agent::TurnTelemetry {
        phase_latencies: hi_agent::TurnPhaseLatencies {
            model_request_ms: 1200,
            tool_batch_ms: 340,
            verify_ms: 2100,
            review_ms: 800,
            finalize_ms: 4,
        },
        effective_max_steps: 120,
        verify_rounds: 2,
        recovery_retries: 1,
        repeat_nudges: 0,
        continue_nudges: 1,
        truncation_retries: 0,
        no_progress_streak: 0,
        forced_final_answer_attempts: 0,
        last_progress_reason: "accepted final answer".to_string(),
        last_stall_reason: String::new(),
        hit_step_cap: false,
        stalled_unfinished: false,
        stalled_repeating: false,
        verify_attributions: Vec::new(),
        verification_executions: Vec::new(),
        tool_calls: 7,
        max_concurrent_batch: 3,
        serial_runs: 2,
        tool_timeline: Vec::new(),
        progress_events: Vec::new(),
        file_reads: 2,
        targeted_searches: 1,
        listing_only: false,
        first_tool_kind: "read".to_string(),
        discovery_depth: "mixed".to_string(),
        quality_repair_nudges: 5,
        review_repair_exhaustion_reason: "review_listing_only_exhausted".to_string(),
        review_repair_counts: repair_counts,
        review_repair_stopped_by_exhaustion: true,
        skeptic_unavailable_count: 0,
        skeptic_last_status: None,
        review_unavailable_reason: None,
        checkpoint_available: None,
        advertised_tools: vec!["read".to_string(), "grep".to_string()],
        tool_schema_tokens: 512,
        prefix_stable_rounds: 6,
        prefix_break_rounds: 1,
        earliest_prefix_break: Some(3),
        ..hi_agent::TurnTelemetry::default()
    });
    app.turn_tool_calls = 7;
    app.apply(UiEvent::Usage {
        prompt: 12,
        generated: 340,
        ctx_used: 64_000,
        ctx_window: Some(128_000),
        estimated: false,
    });
    let mut term = Terminal::new(TestBackend::new(96, 18)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("agent (Ctrl-? to close)"),
        "panel header: {screen}"
    );
    assert!(
        screen.contains("2 verify") && screen.contains("1 retry") && screen.contains("1 continue"),
        "telemetry counters: {screen}"
    );
    assert!(
        screen.contains("tool calls this turn: 7"),
        "tool-call count: {screen}"
    );
    assert!(
        screen.contains("user prompt estimate 12 · output across all model calls 340 · ctx 50%"),
        "scoped token metrics: {screen}"
    );
    assert!(
        screen.contains("review repair: total 5")
            && screen.contains("top listing=4")
            && screen.contains("exhausted listing"),
        "review repair diagnostics: {screen}"
    );

    // Closing drops the panel.
    app.show_debug = false;
    term.draw(|f| app.render(f)).unwrap();
    let screen2 = dump(&term);
    assert!(
        !screen2.contains("agent (Ctrl-? to close)"),
        "panel closed: {screen2}"
    );
}

#[test]
fn ctrl_question_compacts_long_review_repair_mode_names() {
    let mut app = test_app("openai", "gpt-4o");
    app.show_debug = true;
    let mut repair_counts = std::collections::BTreeMap::new();
    repair_counts.insert("review_security_broad_search".to_string(), 12);
    repair_counts.insert("review_gap_search_overclaim".to_string(), 9);
    app.last_telemetry = Some(hi_agent::TurnTelemetry {
        effective_max_steps: 120,
        quality_repair_nudges: 21,
        review_repair_exhaustion_reason: "review_security_broad_search_exhausted".to_string(),
        review_repair_counts: repair_counts,
        review_repair_stopped_by_exhaustion: true,
        ..hi_agent::TurnTelemetry::default()
    });

    let mut term = Terminal::new(TestBackend::new(96, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("top security_broad=12")
            && screen.contains("gap_overclaim=9")
            && screen.contains("exhausted security_broad"),
        "compact review-repair labels: {screen}"
    );
    assert!(
        !screen.contains("review_security_broad_search")
            && !screen.contains("review_gap_search_overclaim"),
        "raw long repair keys should not render in Ctrl-?: {screen}"
    );
}

#[test]
fn in_progress_line_is_styled_live() {
    // A heading still streaming (no trailing newline yet) renders styled with
    // its markers stripped — not literally as "## …" until the line commits.
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Text {
        text: "## Hello world".into(),
    });
    let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Hello world"),
        "heading text shown:\n{screen}"
    );
    assert!(
        !screen.contains("## Hello"),
        "marker stripped live:\n{screen}"
    );

    // Styling the preview must NOT advance the real fence state: a partial
    // opening fence leaves code_lang untouched until its line commits.
    let mut app2 = test_app("openai", "gpt-4o");
    app2.apply(UiEvent::Text {
        text: "```rust".into(),
    });
    term.draw(|f| app2.render(f)).unwrap();
    assert!(
        app2.code_lang.is_none(),
        "live preview must not mutate the committed fence state"
    );
}

#[test]
fn edit_key_submits_on_enter_and_clears() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("queue me");
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(app.edit_key(&enter).as_deref(), Some("queue me"));
    assert!(app.input.is_empty(), "input cleared after submit");
    // An empty Enter submits nothing when idle with no plan and no suggestion.
    assert_eq!(app.edit_key(&enter), None);
}

#[test]
fn empty_enter_submits_suggested_prompt() {
    let mut app = test_app("openai", "gpt-4o");
    app.suggested_prompt = Some("Run the unit tests".into());
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(app.edit_key(&enter).as_deref(), Some("Run the unit tests"));
    assert!(app.suggested_prompt.is_none());
}

#[test]
fn empty_enter_resumes_incomplete_plan() {
    let mut app = test_app("openai", "gpt-4o");
    app.plan = vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.sync_last_drive();
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(
        app.edit_key(&enter).as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );
}

#[test]
fn empty_enter_idle_in_plan_mode_resumes_pause_and_park() {
    let mut app = test_app("openai", "gpt-4o");
    app.plan = vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.plan_mode = true;
    app.sync_last_drive();
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(app.edit_key(&enter), None);

    app.plan_mode = false;
    app.plan_drive_paused = true;
    app.sync_last_drive();
    assert_eq!(
        app.edit_key(&enter).as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );

    app.plan_drive_paused = false;
    app.last_drive = hi_agent::DriveAction::Idle {
        reason: hi_agent::DriveIdleReason::PlanParked,
    };
    assert_eq!(
        app.edit_key(&enter).as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );
}

#[test]
fn empty_enter_submits_drive_after_completed_leftover() {
    let mut app = test_app("openai", "gpt-4o");
    app.plan = vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.last_turn_state = TurnState::Done("verified".into());
    app.last_stop_reason = Some(hi_agent::TurnStopReason::Completed);
    app.sync_last_drive();
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(
        app.edit_key(&enter).as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );
}

#[test]
fn empty_enter_resumes_paused_and_parked_goal() {
    let mut app = test_app("openai", "gpt-4o");
    app.goal = Some(hi_agent::Goal::new("ship it", vec!["implement it".into()]));
    app.goal
        .as_mut()
        .unwrap()
        .pause(hi_agent::GoalPauseReason::User);
    app.sync_last_drive();
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(
        app.edit_key(&enter).as_deref(),
        Some(hi_agent::GOAL_CONTINUE_PROMPT)
    );

    app.goal.as_mut().unwrap().resume();
    app.last_drive = hi_agent::DriveAction::Idle {
        reason: hi_agent::DriveIdleReason::GoalParked,
    };
    assert_eq!(
        app.edit_key(&enter).as_deref(),
        Some(hi_agent::GOAL_CONTINUE_PROMPT)
    );

    app.plan_mode = true;
    app.sync_last_drive();
    assert_eq!(app.edit_key(&enter), None);
}

#[test]
fn drive_chrome_does_not_dump_synthetic_prompt() {
    let line = hi_agent::drive_chrome_line(
        hi_agent::PLAN_DRIVE_PROMPT,
        Some("wire the scheduler"),
        None,
    )
    .expect("plan chrome");
    assert_eq!(line, "⟳ plan drive — wire the scheduler");
    assert!(!line.contains(hi_agent::PLAN_DRIVE_PROMPT));
}

#[test]
fn block_nav_folds_one_block_independently() {
    use crate::TranscriptEntry;
    let mut app = test_app("openai", "gpt-4o");
    let long =
        || -> Vec<Line<'static>> { (0..40).map(|i| Line::raw(format!("line {i}"))).collect() };
    app.transcript.push(TranscriptEntry::ToolOutput {
        body: long(),
        expanded: false,
    });
    app.transcript.push(TranscriptEntry::ToolOutput {
        body: long(),
        expanded: false,
    });
    assert_eq!(app.tool_block_count(), 2);

    // Ctrl-B enters nav on the most recent block.
    app.edit_key(&KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL));
    assert!(app.mode.is_block_nav(), "Ctrl-B enters nav mode");
    assert_eq!(app.selected_block_ord(), 1, "starts on the last block");

    // Enter unfolds just that block.
    app.edit_key(&KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    let expanded: Vec<bool> = app
        .transcript
        .iter()
        .filter_map(|e| match e {
            TranscriptEntry::ToolOutput { expanded, .. } => Some(*expanded),
            _ => None,
        })
        .collect();
    assert_eq!(
        expanded,
        vec![false, true],
        "only the selected block toggled"
    );

    // k/Up moves to the older block; Space toggles it.
    app.edit_key(&KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(app.selected_block_ord(), 0);
    app.edit_key(&KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE));
    let both: Vec<bool> = app
        .transcript
        .iter()
        .filter_map(|e| match e {
            TranscriptEntry::ToolOutput { expanded, .. } => Some(*expanded),
            _ => None,
        })
        .collect();
    assert_eq!(both, vec![true, true]);

    // The cursor never runs past the ends.
    app.edit_key(&KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(app.selected_block_ord(), 0, "clamped at the top");

    // Esc leaves nav mode; keys go back to the input line.
    app.edit_key(&KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(!app.mode.is_block_nav());
    app.edit_key(&KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
    assert_eq!(
        app.input.text(),
        "j",
        "keys type into the input once nav exits"
    );
}

#[test]
fn streamed_table_commits_aligned_after_it_ends() {
    let mut app = test_app("openai", "gpt-4o");
    // Stream a full pipe table. Every line is a table row, so it accumulates in
    // the buffer and nothing is committed yet.
    app.stream(
        ratatui::style::Style::default(),
        true,
        "| A | Long |\n|---|---|\n| x | y |\n",
    );
    assert!(
        app.transcript.is_empty(),
        "table stays buffered until it ends"
    );
    // A following non-table line flushes the table as an aligned block.
    app.stream(ratatui::style::Style::default(), true, "after\n");
    let texts = flatten_texts(&app, false, false);
    assert_eq!(
        texts.len(),
        4,
        "3 aligned rows + the trailing line: {texts:?}"
    );
    assert_eq!(texts[3], "after");
    // Header (row 0) and data (row 2) are padded to the same width.
    assert_eq!(
        texts[0].chars().count(),
        texts[2].chars().count(),
        "columns aligned across rows: {texts:?}"
    );
    assert!(texts[1].starts_with('├'), "ruled separator: {:?}", texts[1]);
}

#[test]
fn streaming_preview_shows_cursor_during_stream_and_clears_after_flush() {
    let mut app = test_app("openai", "gpt-4o");
    // Stream a partial line (no trailing newline) — the pending preview should
    // be live, and the render should show the block cursor.
    app.stream(ratatui::style::Style::default(), true, "hello wor");
    assert!(app.pending.is_some(), "pending line is live mid-stream");
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("▍"),
        "streaming preview shows a block cursor: {screen}"
    );
    assert!(
        screen.contains("hello wor"),
        "partial line text is visible mid-stream: {screen}"
    );
    // Complete the line — the cursor should disappear once flushed.
    app.stream(ratatui::style::Style::default(), true, "ld\n");
    app.flush_pending();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        !screen.contains("▍"),
        "no block cursor after the line is committed: {screen}"
    );
    assert!(
        screen.contains("hello world"),
        "completed line is visible: {screen}"
    );
}

#[test]
fn streamed_code_block_is_captured_for_copy_last_code_block() {
    let mut app = test_app("openai", "gpt-4o");
    // Stream a fenced code block with a language tag and two interior lines.
    app.stream(
        ratatui::style::Style::default(),
        true,
        "```rust\nfn main() {}\nlet x = 1;\n```\n",
    );
    // The last code block should hold the two interior lines (no fence markers).
    assert_eq!(
        app.last_code_block.as_deref(),
        Some("fn main() {}\nlet x = 1;"),
        "interior code lines captured without fence markers"
    );
    // A second block replaces the first as the "last" block.
    app.stream(
        ratatui::style::Style::default(),
        true,
        "```python\nprint('hi')\n```\n",
    );
    assert_eq!(
        app.last_code_block.as_deref(),
        Some("print('hi')"),
        "the most recent block is the one Ctrl-Y copies"
    );
}

#[test]
fn copy_last_code_block_falls_back_to_transcript_scan() {
    let mut app = test_app("openai", "gpt-4o");
    // Simulate a resumed session: transcript has rendered code lines (with the
    // `▏ ` gutter) but `last_code_block` was never populated by streaming.
    app.last_code_block = None;
    // Push a non-code line, then a fenced block as markdown_line would render it.
    app.push(ratatui::text::Line::raw("Here is some code:"));
    // Fence-open line: gutter + language tag.
    app.push(crate::render::markdown_line("```rust", &mut None));
    // Interior code lines.
    let mut lang = Some("rust".to_string());
    app.push(crate::render::markdown_line("fn main() {}", &mut lang));
    app.push(crate::render::markdown_line("let x = 1;", &mut lang));
    // Fence-close line.
    app.push(crate::render::markdown_line("```", &mut lang));
    let block = app.scan_transcript_for_last_code_block();
    assert_eq!(
        block.as_deref(),
        Some("fn main() {}\nlet x = 1;"),
        "fallback scan extracts interior code lines without fence markers"
    );
}

#[test]
fn shell_escape_prefix_runs_command_and_pushes_output() {
    let mut app = test_app("openai", "gpt-4o");
    // Use a workspace root that exists (the crate root) so `sh -c` runs there.
    app.workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    app.run_shell_escape("echo hello-shell-escape");
    // The transcript should contain the `$ echo ...` header and the output line.
    let texts: Vec<String> = app.transcript.iter().map(|e| e.text()).collect();
    let joined = texts.join("\n");
    assert!(
        joined.contains("hello-shell-escape"),
        "shell-escape output should land in the transcript: {joined}"
    );
    assert!(
        joined.contains("$ echo hello-shell-escape"),
        "the command header should be shown: {joined}"
    );
}

#[test]
fn confirmation_modal_colors_file_edit_diff() {
    use hi_agent::ConfirmationRequest;
    let mut app = test_app("openai", "gpt-4o");
    app.confirmation = Some(ConfirmationRequest::FileEdit {
        path: "src/main.rs".to_string(),
        diff: "--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,2 +1,2 @@\n-old\n+new\n ctx\n"
            .to_string(),
    });
    let mut term = Terminal::new(TestBackend::new(80, 32)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    // The diff header and the path should appear; the modal title too.
    assert!(
        screen.contains("Confirm file edit"),
        "modal title shown: {screen}"
    );
    assert!(screen.contains("src/main.rs"), "file path shown: {screen}");
    assert!(screen.contains("+new"), "added diff line shown: {screen}");
    assert!(screen.contains("-old"), "removed diff line shown: {screen}");
}

#[test]
fn ask_user_modal_shows_question_and_options() {
    use hi_agent::ConfirmationRequest;
    let mut app = test_app("openai", "gpt-4o");
    app.confirmation = Some(ConfirmationRequest::AskUser {
        question: "Which transport should the public API use?".into(),
        options: vec!["REST".into(), "gRPC".into()],
    });
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Question for you"),
        "modal title shown: {screen}"
    );
    assert!(
        screen.contains("Which transport"),
        "question shown: {screen}"
    );
    assert!(screen.contains("REST"), "option shown: {screen}");
    assert!(screen.contains("1-9 pick"), "ask_user hint shown: {screen}");
}

#[test]
fn review_overlay_shows_full_diff_with_title() {
    let mut app = test_app("openai", "gpt-4o");
    app.workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    app.mode = crate::mode::UiMode::Review;
    app.diff_text = Some(
        "--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,2 +1,2 @@\n-old\n+new\n ctx\n".to_string(),
    );
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Diff review"),
        "overlay title shown: {screen}"
    );
    assert!(screen.contains("+new"), "added line visible: {screen}");
    assert!(screen.contains("-old"), "removed line visible: {screen}");
    assert!(
        screen.contains("n/p hunks"),
        "keybinding footer shown: {screen}"
    );
}

#[tokio::test]
async fn durable_command_is_tui_first_and_fails_closed_without_a_saved_session() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    let mut app = test_app("openai", "gpt-4o");

    app.handle_command(&mut agent, hi_agent::Command::Durable("on".into()))
        .await;

    assert_eq!(app.execution, hi_agent::ExecutionMode::Ephemeral);
    assert!(
        app.transcript_text()
            .contains("couldn't enable durable execution")
    );
}

#[test]
fn show_session_files_lists_accumulated_files() {
    let mut app = test_app("openai", "gpt-4o");
    // Simulate two turns touching different files.
    app.last_changed_files = vec!["src/main.rs".to_string(), "src/lib.rs".to_string()];
    app.accumulate_session_files();
    app.last_changed_files = vec!["src/lib.rs".to_string(), "src/render.rs".to_string()];
    app.accumulate_session_files();
    // The session set should be deduplicated, first-seen order.
    assert_eq!(
        app.session_changed_files,
        vec![
            "src/main.rs".to_string(),
            "src/lib.rs".to_string(),
            "src/render.rs".to_string(),
        ],
        "session files accumulate and dedupe across turns"
    );
    // `/files` should render the list into the transcript.
    app.show_session_files();
    let text = app.transcript_text();
    assert!(
        text.contains("3 files changed this session"),
        "header shows count: {text}"
    );
    assert!(
        text.contains("src/main.rs") && text.contains("src/render.rs"),
        "file paths listed: {text}"
    );
}

#[test]
fn show_session_files_handles_empty_session() {
    let mut app = test_app("openai", "gpt-4o");
    app.show_session_files();
    let text = app.transcript_text();
    assert!(
        text.contains("no files changed this session yet"),
        "empty session message: {text}"
    );
}

#[test]
fn normal_mode_renders_banner_and_hides_cursor() {
    let mut app = test_app("openai", "gpt-4o");
    app.mode = crate::mode::UiMode::Normal { search: None };
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("-- NORMAL --"),
        "normal-mode banner shown: {screen}"
    );
    assert!(
        screen.contains("j/k scroll"),
        "keybinding hint shown: {screen}"
    );
}

#[test]
fn normal_mode_search_banner_shows_query() {
    let mut app = test_app("openai", "gpt-4o");
    app.mode = crate::mode::UiMode::Normal { search: None };
    app.mode = crate::mode::UiMode::Normal {
        search: Some("render".to_string()),
    };
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("-- SEARCH --"),
        "search banner shown: {screen}"
    );
    assert!(screen.contains("/render"), "search query shown: {screen}");
}

#[test]
fn search_transcript_finds_and_scrolls_to_match() {
    let mut app = test_app("openai", "gpt-4o");
    // Push enough lines that the transcript overflows a 24-row terminal, so
    // scrolling to a match is meaningful (view_max_scroll > 0).
    for i in 0..30 {
        app.push(ratatui::text::Line::raw(format!("filler line {i}")));
    }
    app.push(ratatui::text::Line::raw("the render function"));
    for i in 0..10 {
        app.push(ratatui::text::Line::raw(format!("more filler {i}")));
    }
    app.push(ratatui::text::Line::raw("another render here"));
    // view_max_scroll is normally computed during render; set it manually so
    // scroll_to doesn't clamp to 0 in this unit test (no render happens).
    app.view_max_scroll = 50;
    // Compute the transcript text to find the expected line index.
    let text = app.transcript_text();
    let lines: Vec<&str> = text.lines().collect();
    let first_render = lines
        .iter()
        .position(|l| l.contains("render"))
        .expect("first render match exists");
    // Search forward for "render" from the top — should scroll to the first match.
    search_transcript(&mut app, "render", 1);
    assert_eq!(
        app.scroll as usize, first_render,
        "search should scroll to the first match at line {first_render}, got {}",
        app.scroll
    );
    // Search forward again — should advance to the next "render".
    let second_render = lines
        .iter()
        .rposition(|l| l.contains("render"))
        .expect("second render match exists");
    search_transcript(&mut app, "render", 1);
    assert_eq!(
        app.scroll as usize, second_render,
        "n should advance to the next match at line {second_render}"
    );
    // Search backward — should go back to the first match.
    search_transcript(&mut app, "render", -1);
    assert_eq!(
        app.scroll as usize, first_render,
        "N should return to the previous match"
    );
}

#[test]
fn scroll_to_top_and_bottom_set_following_correctly() {
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..50 {
        app.push(ratatui::text::Line::raw(format!("line {i}")));
    }
    // Scroll to top: following=false, scroll=0.
    app.scroll_to_top();
    assert!(!app.following, "scroll_to_top stops following");
    assert_eq!(app.scroll, 0, "scroll_to_top sets scroll to 0");
    // Scroll to bottom: following=true.
    app.scroll_to_bottom();
    assert!(app.following, "scroll_to_bottom resumes following");
}

#[test]
fn review_next_hunk_jumps_between_hunk_headers() {
    let diff = "diff --git a/foo b/foo\n\
                --- a/foo\n\
                +++ b/foo\n\
                @@ -1,1 +1,1 @@\n\
                -a\n\
                +b\n\
                @@ -5,1 +5,1 @@\n\
                -c\n\
                +d\n\
                @@ -10,1 +10,1 @@\n\
                -e\n\
                +f\n";
    // From line 0 (before first hunk), n → first hunk at line 3.
    assert_eq!(review_next_hunk(Some(diff), 0, 1), 3);
    // From line 3 (first hunk), n → second hunk at line 6.
    assert_eq!(review_next_hunk(Some(diff), 3, 1), 6);
    // From line 6 (second hunk), n → third hunk at line 9.
    assert_eq!(review_next_hunk(Some(diff), 6, 1), 9);
    // From line 9 (third hunk), n → clamps to last line (no more hunks).
    assert_eq!(review_next_hunk(Some(diff), 9, 1), 11);
    // From line 9, p → previous hunk at line 6.
    assert_eq!(review_next_hunk(Some(diff), 9, -1), 6);
    // From line 6, p → previous hunk at line 3.
    assert_eq!(review_next_hunk(Some(diff), 6, -1), 3);
    // From line 3, p → clamps to 0 (no earlier hunk).
    assert_eq!(review_next_hunk(Some(diff), 3, -1), 0);
    // None diff → returns `from` unchanged.
    assert_eq!(review_next_hunk(None, 5, 1), 5);
}

#[test]
fn open_review_with_no_files_shows_full_diff() {
    let mut app = test_app("openai", "gpt-4o");
    app.workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    app.open_review(None);
    assert!(app.mode.is_review(), "review overlay opened");
    assert!(app.review_scroll == 0, "scroll reset to top");
}

#[test]
fn external_editor_reads_back_edited_text() {
    // Create a tiny "editor" script that appends to its last argument.
    let script = std::env::temp_dir().join(format!(".hi-test-editor-{}", std::process::id()));
    std::fs::write(&script, "#!/bin/sh\nprintf 'edited' >> \"$1\"\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();
    }
    unsafe {
        std::env::set_var("VISUAL", "");
        std::env::set_var("EDITOR", script.to_str().unwrap());
        std::env::set_var("HI_TUI_NO_TERMINAL", "1");
    }
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("original prompt");
    app.edit_in_external_editor();
    let text = app.input.text();
    assert!(
        text.contains("original prompt") && text.contains("edited"),
        "input should contain the edited text: {text}"
    );
    // Clean up.
    unsafe {
        std::env::remove_var("EDITOR");
        std::env::remove_var("HI_TUI_NO_TERMINAL");
    }
    let _ = std::fs::remove_file(&script);
}

#[test]
fn mouse_drag_selects_a_line_range_and_keeps_it() {
    use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..5 {
        app.transcript
            .push(crate::TranscriptEntry::Line(Line::raw(format!("row {i}"))));
    }
    // Geometry the render pass would cache: inner rect at (1,1), no scroll, each
    // line exactly one wrapped row.
    app.view_inner = ratatui::layout::Rect {
        x: 1,
        y: 1,
        width: 80,
        height: 10,
    };
    app.view_scroll = 0;
    app.view_prefix = vec![0, 1, 2, 3, 4, 5];
    app.view_line_texts = (0..5).map(|i| format!("row {i}")).collect();

    let ev = |kind, col, row| MouseEvent {
        kind,
        column: col,
        row,
        modifiers: KeyModifiers::NONE,
    };
    // Press on line 1 (screen row 2 → abs 1); no selection range change yet.
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 5, 2));
    assert_eq!(app.selection_range(), Some((1, 1)));
    assert!(!app.select_dragged);
    // Drag down to line 3 (screen row 4 → abs 3).
    app.handle_mouse(ev(MouseEventKind::Drag(MouseButton::Left), 5, 4));
    assert_eq!(app.selection_range(), Some((1, 3)));
    assert!(app.select_dragged);
    // The exact text a release would copy (pure — no real clipboard touched).
    assert_eq!(app.selected_text().as_deref(), Some("row 1\nrow 2\nrow 3"));
    // A drag that runs off the bottom edge clamps to the last visible line.
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 5, 3)); // abs 2
    app.handle_mouse(ev(MouseEventKind::Drag(MouseButton::Left), 5, 250));
    assert_eq!(
        app.selection_range(),
        Some((2, 4)),
        "clamped to the last line"
    );
}

#[test]
fn mouse_drag_within_one_line_selects_characters() {
    use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
    let mut app = test_app("openai", "gpt-4o");
    app.transcript.push(crate::TranscriptEntry::Line(Line::raw(
        "hello world foobar",
    )));
    app.view_inner = ratatui::layout::Rect {
        x: 1,
        y: 1,
        width: 80,
        height: 10,
    };
    app.view_scroll = 0;
    app.view_prefix = vec![0, 1]; // one logical line, one display row
    app.view_line_texts = vec!["hello world foobar".to_string()];
    let ev = |kind, col, row| MouseEvent {
        kind,
        column: col,
        row,
        modifiers: KeyModifiers::NONE,
    };
    // 'w' of "world" is char index 6 → screen col = inner.x(1) + 6 = 7; drag to
    // just past 'd' (index 11) → col 12. The character range 6..11 is "world".
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 7, 1));
    app.handle_mouse(ev(MouseEventKind::Drag(MouseButton::Left), 12, 1));
    assert_eq!(app.char_span(), Some((0, 6, 11)), "single-line char range");
    assert_eq!(app.selected_text().as_deref(), Some("world"));

    // Extending across a second line falls back to whole-line selection.
    app.transcript
        .push(crate::TranscriptEntry::Line(Line::raw("second line")));
    app.view_prefix = vec![0, 1, 2];
    app.view_line_texts = vec!["hello world foobar".into(), "second line".into()];
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 7, 1)); // line 0
    app.handle_mouse(ev(MouseEventKind::Drag(MouseButton::Left), 5, 2)); // line 1
    assert_eq!(app.char_span(), None, "multi-line → no char span");
    assert_eq!(
        app.selected_text().as_deref(),
        Some("hello world foobar\nsecond line")
    );
}

#[test]
fn mouse_up_without_drag_events_still_selects_and_copies() {
    // Terminals that omit intermediate Drag and only emit Down + Up at
    // different cells must still extend the selection and auto-copy.
    use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..5 {
        app.transcript
            .push(crate::TranscriptEntry::Line(Line::raw(format!("row {i}"))));
    }
    app.view_inner = ratatui::layout::Rect {
        x: 1,
        y: 1,
        width: 80,
        height: 10,
    };
    app.view_scroll = 0;
    app.view_prefix = vec![0, 1, 2, 3, 4, 5];
    app.view_line_texts = (0..5).map(|i| format!("row {i}")).collect();
    let ev = |kind, col, row| MouseEvent {
        kind,
        column: col,
        row,
        modifiers: KeyModifiers::NONE,
    };
    // Press line 1, release on line 3 — no Drag in between.
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 5, 2));
    assert!(!app.select_dragged);
    app.handle_mouse(ev(MouseEventKind::Up(MouseButton::Left), 5, 4));
    assert!(app.select_dragged, "release at a new cell counts as a drag");
    assert_eq!(app.selection_range(), Some((1, 3)));
    assert_eq!(app.selected_text().as_deref(), Some("row 1\nrow 2\nrow 3"));
    // copy_selection ran; toast is set on success (clipboard may fail in CI —
    // either toast or a "copy failed" line is fine; selection must remain).
    assert!(
        app.copy_toast.is_some()
            || app
                .transcript
                .iter()
                .any(|e| e.text().contains("copy failed")),
        "release after motion must attempt auto-copy"
    );
}

#[test]
fn whole_line_selection_strips_display_gutters() {
    use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
    let mut app = test_app("openai", "gpt-4o");
    app.view_inner = ratatui::layout::Rect {
        x: 1,
        y: 1,
        width: 80,
        height: 10,
    };
    app.view_scroll = 0;
    app.view_prefix = vec![0, 1, 2];
    // Painted lines include the tool / code gutters.
    app.view_line_texts = vec!["┃ tool output".into(), "▏ code body".into()];
    let ev = |kind, col, row| MouseEvent {
        kind,
        column: col,
        row,
        modifiers: KeyModifiers::NONE,
    };
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 5, 1));
    app.handle_mouse(ev(MouseEventKind::Drag(MouseButton::Left), 5, 2));
    assert_eq!(
        app.selected_text().as_deref(),
        Some("tool output\ncode body"),
        "whole-line copy must drop decorative gutters"
    );
}

#[test]
fn mouse_plain_click_folds_and_leaves_no_selection() {
    use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
    let mut app = test_app("openai", "gpt-4o");
    let body: Vec<Line<'static>> = (0..40).map(|i| Line::raw(format!("l{i}"))).collect();
    app.transcript.push(crate::TranscriptEntry::ToolOutput {
        body,
        expanded: false,
    });
    app.view_inner = ratatui::layout::Rect {
        x: 1,
        y: 1,
        width: 80,
        height: 20,
    };
    app.view_scroll = 0;
    app.view_prefix = (0..=40).collect();
    app.view_line_texts = (0..40).map(|i| format!("l{i}")).collect();
    app.block_row_spans = vec![(0, 17, 0)];

    let ev = |kind, col, row| MouseEvent {
        kind,
        column: col,
        row,
        modifiers: KeyModifiers::NONE,
    };
    // Down then Up at the same spot (no drag) → a fold, not a selection.
    app.handle_mouse(ev(MouseEventKind::Down(MouseButton::Left), 5, 3));
    app.handle_mouse(ev(MouseEventKind::Up(MouseButton::Left), 5, 3));
    assert!(app.selection_range().is_none(), "click leaves no selection");
    let expanded = match &app.transcript[0] {
        crate::TranscriptEntry::ToolOutput { expanded, .. } => *expanded,
        _ => unreachable!(),
    };
    assert!(expanded, "the clicked block folded open");
}

#[test]
fn mouse_click_folds_the_block_under_it() {
    use crate::TranscriptEntry;
    let mut app = test_app("openai", "gpt-4o");
    let long = || -> Vec<Line<'static>> { (0..40).map(|i| Line::raw(format!("l{i}"))).collect() };
    app.transcript.push(TranscriptEntry::ToolOutput {
        body: long(),
        expanded: false,
    });
    app.transcript.push(TranscriptEntry::ToolOutput {
        body: long(),
        expanded: false,
    });
    // Simulate the geometry the render pass caches: inner area at (1,1), no
    // scroll, block 0 spanning wrapped rows 0..17 and block 1 rows 17..34.
    app.view_inner = ratatui::layout::Rect {
        x: 1,
        y: 1,
        width: 80,
        height: 20,
    };
    app.view_scroll = 0;
    app.block_row_spans = vec![(0, 17, 0), (17, 34, 1)];

    let expanded = |app: &crate::App| -> Vec<bool> {
        app.transcript
            .iter()
            .filter_map(|e| match e {
                TranscriptEntry::ToolOutput { expanded, .. } => Some(*expanded),
                _ => None,
            })
            .collect()
    };

    // Screen row 20 → abs row 19 ∈ [17,34) → block 1.
    app.handle_click(5, 20);
    assert_eq!(expanded(&app), vec![false, true], "clicked block toggled");
    assert_eq!(app.block_cursor, 1, "cursor moved to the clicked block");

    // A click below the transcript area is ignored.
    app.handle_click(5, 100);
    assert_eq!(
        expanded(&app),
        vec![false, true],
        "out-of-area click ignored"
    );

    // Screen row 2 → abs row 1 ∈ [0,17) → block 0 toggles open too.
    app.handle_click(5, 2);
    assert_eq!(expanded(&app), vec![true, true]);
}

#[test]
fn block_nav_expanded_block_shows_full_body() {
    use crate::TranscriptEntry;
    let body: Vec<Line<'static>> = (0..40).map(|i| Line::raw(format!("l{i}"))).collect();
    // Folded (default): a preview plus a fold footer.
    let folded = TranscriptEntry::ToolOutput {
        body: body.clone(),
        expanded: false,
    };
    let flat = folded.flatten(false, false, crate::Density::Comfortable);
    assert!(flat.len() < 40, "folded to a preview: {} lines", flat.len());
    assert!(
        flat.iter()
            .any(|l| crate::render::line_text(l).contains("more lines")),
        "fold footer present"
    );
    // Per-block expand shows the whole body without the global toggle.
    let open = TranscriptEntry::ToolOutput {
        body,
        expanded: true,
    };
    let flat = open.flatten(false, false, crate::Density::Comfortable);
    assert!(
        flat.len() >= 40,
        "expanded shows full body: {} lines",
        flat.len()
    );
}

#[test]
fn renders_title_transcript_and_input() {
    let mut app = test_app("openai", "gpt-4o");
    app.push(Line::raw("› hello"));
    app.apply(UiEvent::Text {
        text: "hi there\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    app.input.set("next question");

    let mut term = Terminal::new(TestBackend::new(50, 12)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    assert!(screen.contains("gpt-4o"), "title shows model");
    assert!(screen.contains("hello"), "user line");
    assert!(screen.contains("hi there"), "assistant line");
    assert!(screen.contains("next question"), "input box");
}

#[test]
fn responsive_session_layouts_keep_the_composer_inside_the_screen() {
    for (width, height) in [(120, 24), (80, 20), (64, 14), (48, 12), (40, 10), (24, 8)] {
        let mut app = test_app("openai", "gpt-4o");
        app.push_user_prompt(Line::raw("❯ review the responsive layout"));
        app.apply(UiEvent::Text {
            text: "The layout remains usable while the terminal is narrow.\n".into(),
        });
        app.apply(UiEvent::AssistantEnd);
        app.input.set("next question");

        let mut term = Terminal::new(TestBackend::new(width, height)).unwrap();
        term.draw(|frame| app.render(frame)).unwrap();
        let screen = dump(&term);
        let cursor = term.backend().cursor_position();
        assert!(
            cursor.x < width && cursor.y < height,
            "cursor outside {width}x{height}: {cursor:?}\n{screen}"
        );
        assert!(
            screen.contains("gpt-4o"),
            "model identity at {width}x{height}: {screen}"
        );
        assert!(
            screen.contains("next question"),
            "input at {width}x{height}: {screen}"
        );
        assert!(
            screen
                .lines()
                .any(|line| line.trim_start().starts_with('╰')),
            "composer border closes at {width}x{height}: {screen}"
        );
    }
}

#[test]
fn btw_overlay_stays_visible_across_resize() {
    let mut app = test_app("openai", "gpt-4o");
    app.show_btw = true;
    app.btw_thread.push(BtwEntry::Question("why?".into()));
    app.btw_thread.push(BtwEntry::Thinking("answering…".into()));

    for (width, height) in [(120, 20), (64, 14)] {
        let mut term = Terminal::new(TestBackend::new(width, height)).unwrap();
        term.draw(|frame| app.render(frame)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("/btw why?"),
            "overlay title at {width}x{height}: {screen}"
        );
        assert!(
            screen.contains("[Esc]"),
            "close hint at {width}x{height}: {screen}"
        );
        assert!(
            screen.contains("Answering"),
            "loading body at {width}x{height}: {screen}"
        );
    }
}

#[test]
fn btw_overlay_measures_prompt_full_width() {
    let mut app = test_app("openai", "gpt-4o");
    app.show_btw = true;
    app.btw_thread.push(BtwEntry::Question("why?".into()));
    app.input
        .set("a deliberately long prompt that must wrap in the main column END");

    let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = dump(&terminal);
    let cursor = terminal.backend().cursor_position();
    assert!(screen.contains("END"), "prompt tail was clipped:\n{screen}");
    assert!(
        cursor.x < 79 && cursor.y < 12,
        "cursor outside composer: {cursor:?}"
    );
    assert!(
        screen
            .lines()
            .any(|line| line.trim_start().starts_with('╰')),
        "composer border must close:\n{screen}"
    );
}

#[test]
fn provider_form_keeps_last_field_cursor_inside_tiny_terminal() {
    let mut app = test_app("openai", "gpt-4o");
    let mut form = crate::provider_form::ProviderForm::new_add();
    form.next_field();
    form.next_field();
    form.next_field();
    app.provider_form = Some(form);

    let mut terminal = Terminal::new(TestBackend::new(24, 8)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let cursor = terminal.backend().cursor_position();
    let screen = dump(&terminal);
    assert!(
        cursor.x < 23 && cursor.y < 7,
        "cursor on/outside border: {cursor:?}\n{screen}"
    );
    assert!(
        screen.contains("Base URL"),
        "active field is visible:\n{screen}"
    );
    assert!(
        screen
            .lines()
            .any(|line| line.trim_start().starts_with('╰')),
        "provider form border must close:\n{screen}"
    );
}

#[test]
fn session_render_snapshots_cover_responsive_chrome() {
    let mut snapshots = String::new();
    for (width, height) in [(120, 24), (64, 14), (24, 8)] {
        let mut app = test_app("openai", "gpt-4o");
        app.push_user_prompt(Line::raw("review the responsive layout"));
        app.page_flip_on_send = false;
        app.following = true;
        app.transcript.push(TranscriptEntry::Assistant(Line::raw(
            "The layout is stable.",
        )));
        app.transcript.push(TranscriptEntry::Reasoning {
            text: "Check spacing and preserve the active input.".into(),
            elapsed: Duration::from_secs(3),
        });
        app.transcript.push(TranscriptEntry::ToolOutput {
            body: (1..=17)
                .map(|n| Line::raw(format!("tool result line {n}")))
                .collect(),
            expanded: false,
        });
        app.status = "working".into();
        app.working = true;
        app.show_btw = true;
        app.btw_thread.push(BtwEntry::Question("why?".into()));
        app.input.insert_str("next step");

        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        snapshots.push_str(&format!("--- {width}x{height} ---\n"));
        snapshots.push_str(&snapshot_dump(&terminal));
    }

    if std::env::var_os("DUMP_TUI_SNAPSHOTS").is_some() {
        println!("{snapshots}");
        return;
    }
    assert_eq!(
        snapshots,
        include_str!("../snapshots/session_responsive.txt"),
        "session responsive snapshot changed; set DUMP_TUI_SNAPSHOTS=1 to inspect intentionally"
    );
}

#[test]
fn user_prompt_timestamp_sits_on_the_right() {
    let mut app = test_app("openai", "gpt-4o");
    app.timestamps_enabled = true;
    app.push_user_prompt(Line::raw("❯ hello"));
    if let Some(crate::TranscriptEntry::UserPrompt { at, .. }) = app.transcript.last_mut() {
        *at = std::time::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    }
    let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("❯ hello"), "{screen}");
    assert!(
        screen.contains("AM") || screen.contains("PM"),
        "prompt timestamp missing: {screen}"
    );
}

#[test]
fn turn_end_writes_worked_for_into_the_transcript() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.started = Some(std::time::Instant::now() - Duration::from_secs(125));
    app.last_turn_state = TurnState::Done("done".into());
    app.set_working(false);
    let text = app.transcript_text();
    assert!(
        text.contains("Worked for 2m5s"),
        "grok-build turn marker missing: {text}"
    );
}

#[test]
fn session_density_snapshots_keep_fold_contract() {
    let mut app = test_app("openai", "gpt-4o");
    app.transcript.push(TranscriptEntry::ToolOutput {
        body: (1..=17)
            .map(|n| Line::raw(format!("tool result line {n}")))
            .collect(),
        expanded: false,
    });

    let compact = app.transcript[0].flatten(false, false, Density::Compact);
    let comfortable = app.transcript[0].flatten(false, false, Density::Comfortable);
    let verbose = app.transcript[0].flatten(false, false, Density::Verbose);
    assert_eq!(compact.len(), 1);
    assert!(compact[0].to_string().contains("folded"));
    assert_eq!(comfortable.len(), TOOL_OUTPUT_PREVIEW_LINES + 1);
    assert_eq!(verbose.len(), 17);
}

#[test]
fn transcript_roles_get_display_gutters_without_polluting_copy_text() {
    let mut app = test_app("openai", "gpt-4o");
    app.push_user_prompt(Line::raw("❯ question"));
    app.apply(UiEvent::Reasoning {
        text: "checking the approach".into(),
    });
    app.apply(UiEvent::Text {
        text: "answer\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    app.transcript.push(TranscriptEntry::ToolOutput {
        body: vec![Line::raw("tool output")],
        expanded: true,
    });

    let prompt = app.transcript[0].flatten(false, false, Density::Comfortable);
    assert!(crate::render::line_text(&prompt[0]).starts_with("❯"));
    assert_eq!(app.transcript[0].text(), "❯ question");

    let assistant = app
        .transcript
        .iter()
        .find(|entry| {
            matches!(
                entry,
                TranscriptEntry::Assistant(_) | TranscriptEntry::AssistantMessage { .. }
            )
        })
        .unwrap();
    let assistant_lines = assistant.flatten(false, false, Density::Comfortable);
    assert_eq!(crate::render::line_text(&assistant_lines[0]), "answer");
    assert_eq!(assistant.text().trim(), "answer");

    let reasoning = app
        .transcript
        .iter()
        .find(|entry| matches!(entry, TranscriptEntry::Reasoning { .. }))
        .unwrap();
    let thought =
        crate::render::line_text(&reasoning.flatten(true, false, Density::Comfortable)[0]);
    assert!(
        thought.contains("thought") || thought.contains("Thinking"),
        "{thought}"
    );

    let tool = app.transcript.last().unwrap();
    assert!(
        crate::render::line_text(&tool.flatten(false, true, Density::Verbose)[0])
            .starts_with("┃ tool output")
    );
    assert_eq!(tool.text(), "tool output");

    let status = TranscriptEntry::Line(crate::render::accent_line(
        crate::theme::theme().accent_system,
        "status text",
        crate::render::dim(),
    ));
    assert_eq!(status.text(), "status text");
}

#[test]
fn chrome_tones_follow_each_palette() {
    use crate::theme::{Theme, UiTone};

    for palette in [Theme::dark(), Theme::light(), Theme::ansi()] {
        assert_eq!(
            palette.chrome(UiTone::Success).border.fg,
            Some(palette.accent_success)
        );
        assert_eq!(
            palette.chrome(UiTone::Error).border.fg,
            Some(palette.accent_error)
        );
        assert_eq!(
            palette.chrome(UiTone::Active).selected.bg,
            Some(palette.selection_bg)
        );
    }
}

fn turn_outcome(
    status: hi_agent::TurnStatus,
    verification: hi_agent::VerificationStatus,
    review: hi_agent::ReviewStatus,
    stop_reason: hi_agent::TurnStopReason,
) -> hi_agent::TurnOutcome {
    hi_agent::TurnOutcome {
        status,
        verification,
        review,
        stop_reason,
        changed_files: vec!["src/lib.rs".to_string()],
        verified_workspace_revision: (verification == hi_agent::VerificationStatus::Passed)
            .then(|| "revision-1".to_string()),
        effective_route: hi_agent::EffectiveModelRoute {
            provider: Some("test".to_string()),
            model: "model".to_string(),
        },
        review_same_model: false,
        leftover: None,
        plan_leftover: None,
    }
}

#[test]
fn turn_end_is_neutral_until_typed_pass_arrives() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.apply(UiEvent::TurnEnd {
        summary: "[10 in · 2 out · 12 total]".into(),
    });

    assert_eq!(app.last_turn_state, TurnState::Running);
    assert!(
        app.transcript
            .iter()
            .all(|e| !e.text().contains("usage") && !e.text().contains("✓")),
        "turn_end is not a transcript receipt: {:?}",
        app.transcript
            .iter()
            .map(TranscriptEntry::text)
            .collect::<Vec<_>>()
    );

    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::Passed,
        hi_agent::ReviewStatus::Passed,
        hi_agent::TurnStopReason::Completed,
    ));
    assert_eq!(
        app.last_turn_state,
        TurnState::Done("verified · reviewed".to_string())
    );
    assert!(app.transcript.last().unwrap().text().contains("✓ done"));
}

#[test]
fn usage_summary_content_cannot_override_typed_outcome() {
    let mut app = test_app("openai", "gpt-4o");
    let noisy = "[user prompt estimate 10 · output across all model calls 2 · ctx 5% (500/10k) · steer: 2 verify · 1 retry]";
    app.apply(UiEvent::TurnEnd {
        summary: noisy.into(),
    });
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Incomplete,
        hi_agent::VerificationStatus::Unverified,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::VerificationUnavailable,
    ));

    assert!(matches!(app.last_turn_state, TurnState::Warning(_)));
    let transcript = app.transcript_text();
    assert!(
        !transcript.contains("user prompt estimate"),
        "usage stays out of the pane: {transcript}"
    );
    assert!(transcript.contains("⚠ incomplete · checks did not settle"));
    assert!(!transcript.contains("✓ done"));
}

#[test]
fn unverified_completed_mutation_is_warning_not_done() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::Unverified,
        hi_agent::ReviewStatus::NotRequired,
        hi_agent::TurnStopReason::VerificationUnavailable,
    ));

    assert_eq!(
        app.last_turn_state,
        TurnState::Warning("checks did not settle".to_string())
    );
    assert!(!app.transcript_text().contains("✓ done"));
}

#[test]
fn deterministic_pass_survives_review_unavailability() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::Passed,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::Completed,
    ));

    assert_eq!(app.last_turn_state, TurnState::Done("verified".to_string()));
}

#[test]
fn stalled_turn_with_deterministic_pass_is_successful() {
    // A repeat/no-progress guard can fire after the edit is made, but the
    // final deterministic check is authoritative for the settled workspace.
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::Passed,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::Stalled,
    ));

    assert_eq!(app.last_turn_state, TurnState::Done("verified".to_string()));
    assert!(app.transcript_text().contains("✓ done · verified"));
    assert!(!app.transcript_text().contains("review unavailable"));
}

#[test]
fn legacy_incomplete_green_outcome_renders_as_stalled() {
    // Defense in depth for a caller that still supplies the old contradictory
    // combination: Incomplete must never render as a successful turn.
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Incomplete,
        hi_agent::VerificationStatus::Passed,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::Stalled,
    ));

    assert_eq!(
        app.last_turn_state,
        TurnState::Warning("incomplete · stalled".to_string())
    );
    let transcript = app.transcript_text();
    assert!(transcript.contains("⚠ incomplete · stalled"));
    assert!(!transcript.contains("✓ done"));
}

#[test]
fn no_change_stall_cannot_render_done_despite_baseline_pass() {
    let mut app = test_app("openai", "gpt-4o");
    let mut outcome = turn_outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::Passed,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::Stalled,
    );
    outcome.changed_files.clear();
    app.note_turn_outcome(&outcome);

    assert_eq!(
        app.last_turn_state,
        TurnState::Warning("stalled".to_string())
    );
    assert!(app.transcript_text().contains("⚠ stalled"));
    assert!(!app.transcript_text().contains("✓ done"));
}

#[test]
fn review_objection_cannot_render_done() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Incomplete,
        hi_agent::VerificationStatus::Passed,
        hi_agent::ReviewStatus::Objected,
        hi_agent::TurnStopReason::ReviewObjected,
    ));

    assert_eq!(
        app.last_turn_state,
        TurnState::Warning("incomplete · review objected".to_string())
    );
    assert!(!app.transcript_text().contains("✓ done"));
}

#[test]
fn verification_infrastructure_failure_is_failed() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Failed,
        hi_agent::VerificationStatus::InfrastructureError,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::InfrastructureFailure,
    ));

    assert_eq!(
        app.last_turn_state,
        TurnState::Failed("infrastructure failure".to_string())
    );
    // Internal state stays Failed for reports/eval, but the jargon banner is
    // not shown in the user-facing transcript.
    assert!(
        !app.transcript_text().contains("✗ failed"),
        "infrastructure failure must not print a user-facing failure banner"
    );
    assert!(!app.transcript_text().contains("infrastructure failure"));
}

#[test]
fn typed_cancellation_is_cancelled() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Cancelled,
        hi_agent::VerificationStatus::Unverified,
        hi_agent::ReviewStatus::Unavailable,
        hi_agent::TurnStopReason::Cancelled,
    ));

    assert_eq!(app.last_turn_state, TurnState::Cancelled);
    assert!(!app.transcript_text().contains("✓ done"));
}

#[test]
fn not_applicable_checks_stay_out_of_the_transcript() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::NotApplicable,
        hi_agent::ReviewStatus::NotRequired,
        hi_agent::TurnStopReason::NoApplicableVerification,
    ));
    assert_eq!(
        app.last_turn_state,
        TurnState::Done("no applicable checks".to_string())
    );
    let text = app.transcript_text();
    assert!(!text.contains("✓ done"), "non-event painted done: {text}");
    assert!(
        !text.contains("no applicable checks"),
        "non-event leaked into the pane: {text}"
    );
}

#[test]
fn idle_done_does_not_paint_a_status_receipt() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::Passed,
        hi_agent::ReviewStatus::Passed,
        hi_agent::TurnStopReason::Completed,
    ));
    app.last_turn_latency = Some(Duration::from_secs(65));
    app.context_used = 38_000;
    app.context_window = Some(1_000_000);
    let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        !screen.contains("last: done"),
        "done status row should hide: {screen}"
    );
    assert!(
        !screen.contains("usage ·"),
        "usage receipt should stay out of the pane: {screen}"
    );
    assert!(screen.contains("38k / 1.0M"), "ctx chip missing: {screen}");
    assert!(
        !screen.contains("1m 05s"),
        "turn duration belongs in Worked for, not the header: {screen}"
    );
}

#[test]
fn markdown_headings_gain_space_in_the_transcript() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Text {
        text: "intro\n## Section\nbody".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    let texts: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false, crate::Density::Comfortable))
        .map(|line| crate::render::line_text(&line))
        .collect();
    assert_eq!(
        texts,
        vec![
            "intro".to_string(),
            String::new(),
            "Section".to_string(),
            String::new(),
            "body".to_string()
        ],
        "heading should be preceded by a blank row: {texts:?}"
    );
}

#[test]
fn assistant_text_becomes_copy_target() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Text {
        text: "first ".into(),
    });
    app.apply(UiEvent::Text {
        text: "answer\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    assert_eq!(app.last_assistant, "first answer");

    app.apply(UiEvent::ToolCall {
        name: "bash".into(),
        arguments: "{\"command\":\"echo noisy\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "noisy output".into(),
    });
    assert_eq!(
        app.last_assistant, "first answer",
        "tool logs are not copied as the assistant response"
    );
}

#[test]
fn transcript_text_serializes_lines() {
    let mut app = test_app("openai", "gpt-4o");
    app.push(Line::raw("one"));
    app.push(Line::from(vec![Span::raw("t"), Span::raw("wo")]));
    assert_eq!(app.transcript_text(), "one\ntwo");
}

#[test]
fn btw_answer_goes_to_overlay_not_task_answer() {
    let mut app = test_app("openai", "gpt-4o");
    // Main task answer streams normally…
    app.apply(UiEvent::Text {
        text: "the task result".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    // Keystroke path: question lands in the pane immediately with a thinking marker.
    app.btw_note_question("what step?");
    assert!(app.show_btw, "first btw activity opens the overlay");
    assert!(
        app.btw_thread
            .iter()
            .any(|e| matches!(e, crate::BtwEntry::Thinking(_))),
        "in-flight marker so the overlay doesn't look frozen"
    );
    // Agent drain path (may re-send the same question — must not duplicate).
    app.apply(UiEvent::BtwQuestion {
        question: "what step?".into(),
    });
    app.apply(UiEvent::BtwAnswer {
        text: "you're on step 2".into(),
    });
    app.apply(UiEvent::BtwEnd);

    let thread: String = app
        .btw_thread
        .iter()
        .flat_map(|e| e.as_lines())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        thread.contains("what step?") && thread.contains("you're on step 2"),
        "overlay thread holds Q+A, got: {thread:?}"
    );
    assert!(
        !app.btw_thread
            .iter()
            .any(|e| matches!(e, crate::BtwEntry::Thinking(_))),
        "thinking marker clears when the answer finishes"
    );
    let q_count = app
        .btw_thread
        .iter()
        .filter(|e| matches!(e, crate::BtwEntry::Question(q) if q == "what step?"))
        .count();
    assert_eq!(q_count, 1, "question must not be duplicated: {thread:?}");
    let transcript = app.transcript_text();
    // Side channel is overlay-only — main work log stays clean until Esc.
    assert!(
        !transcript.contains("what step?") && !transcript.contains("you're on step 2"),
        "btw must not pollute main transcript: {transcript:?}"
    );
    // The side-answer is NOT folded into the task answer `/copy` would return.
    assert_eq!(app.last_assistant, "the task result");
}

#[test]
fn btw_answer_stays_in_overlay_not_inline_transcript() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::BtwAnswer {
        text: "aside".into(),
    });
    assert!(
        app.btw_thread
            .iter()
            .any(|e| matches!(e, crate::BtwEntry::Answer(a) if a.contains("aside"))),
        "answer lands in the overlay thread"
    );
    assert!(
        !app.transcript_text().contains("aside"),
        "answer is not dumped into the main transcript while the overlay is open"
    );
}

#[test]
fn dismissing_btw_overlay_persists_collapsed_block() {
    let mut app = test_app("openai", "gpt-4o");
    app.btw_note_question("what step?");
    app.apply(UiEvent::BtwAnswer {
        text: "you're on step 2".into(),
    });
    app.apply(UiEvent::BtwEnd);
    assert!(app.dismiss_btw_overlay());
    assert!(!app.show_btw);
    assert!(app.btw_thread.is_empty());
    let text = app.transcript_text();
    assert!(
        text.contains("/btw what step?"),
        "dismissed overlay persists the question: {text:?}"
    );
    assert!(
        text.contains("you're on step 2"),
        "full answer is kept for copy/expand: {text:?}"
    );
    let flattened = app.transcript.last().expect("btw block").flatten(
        false,
        false,
        crate::Density::Comfortable,
    );
    assert_eq!(flattened.len(), 1, "collapsed header only: {flattened:?}");
    assert!(crate::render::line_text(&flattened[0]).contains("/btw what step?"));
}

#[test]
fn btw_overlay_renders_above_the_prompt() {
    let mut app = test_app("openai", "gpt-4o");
    app.show_btw = true;
    app.btw_thread.push(BtwEntry::Question("why?".into()));
    app.btw_thread.push(BtwEntry::Thinking("answering…".into()));
    app.input.set("next step");

    let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = dump(&terminal);
    let btw_at = screen.find("/btw why?").expect("overlay title");
    let prompt_at = screen.find('❯').expect("prompt");
    assert!(
        btw_at < prompt_at,
        "overlay must sit above the prompt:\n{screen}"
    );
    assert!(
        !screen.contains("❓"),
        "side-pane question glyph should be gone:\n{screen}"
    );
}

#[test]
fn typed_incomplete_outcome_is_visible_after_tool_output_without_usage() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"src/main.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "19 additions, 3 deletions".into(),
    });
    app.note_turn_outcome(&turn_outcome(
        hi_agent::TurnStatus::Incomplete,
        hi_agent::VerificationStatus::Unverified,
        hi_agent::ReviewStatus::NotRequired,
        hi_agent::TurnStopReason::Stalled,
    ));

    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines
            .iter()
            .any(|line| line.contains("incomplete · stalled")),
        "transcript: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("degraded in-session")),
        "transcript: {lines:?}"
    );
    assert_eq!(app.status, "warning · incomplete · stalled");
}

#[test]
fn leftover_plan_replaces_stalled_incomplete_banner() {
    let mut app = test_app("openai", "gpt-4o");
    let mut outcome = turn_outcome(
        hi_agent::TurnStatus::Incomplete,
        hi_agent::VerificationStatus::Unverified,
        hi_agent::ReviewStatus::NotRequired,
        hi_agent::TurnStopReason::Stalled,
    );
    outcome.leftover = Some("3/9 remaining — wire the scheduler".into());
    app.note_turn_outcome(&outcome);
    assert_eq!(
        app.status,
        "warning · incomplete · 3/9 remaining — wire the scheduler"
    );
}

#[test]
fn failed_turn_is_visible() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_failed("request failed", "request", "wait a moment, then /retry");
    let line = app.transcript.last().unwrap().text();
    assert!(line.contains("✗ failed"), "got: {line}");
    assert!(line.contains("request failed"), "got: {line}");
    assert!(line.contains("request"), "got: {line}");
    assert!(line.contains("💡"), "shows guidance: {line}");
    assert!(
        app.status.contains("request"),
        "status has kind: {}",
        app.status
    );
}

#[test]
fn tool_protocol_failure_does_not_mark_model_degraded() {
    let mut app = test_app("pipenetwork", "pipe/auto-coder");
    let err: anyhow::Error = hi_ai::ProviderError::new(
        hi_ai::ProviderErrorKind::ToolProtocol,
        "model output did not satisfy the tool protocol",
    )
    .into();
    let (kind, guidance) = hi_agent::classify_error(&err);

    app.note_turn_failed(&format!("{err:#}"), kind, guidance);
    if hi_agent::ui::error_counts_as_model_issue(&err) {
        app.record_model_issue();
    }

    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines.iter().any(|line| line.contains("tool_protocol")),
        "transcript: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("degraded in-session")),
        "transcript: {lines:?}"
    );
    assert_eq!(app.model_issues.get("pipe/auto-coder"), None);
}

#[test]
fn route_rejection_failure_does_not_mark_model_degraded() {
    let mut app = test_app("pipenetwork", "pipe/auto-coder");
    let err: anyhow::Error = hi_ai::ProviderError::new(
        hi_ai::ProviderErrorKind::ModelUnavailable,
        "model temporarily unavailable",
    )
    .into();
    let (kind, guidance) = hi_agent::classify_error(&err);

    app.note_turn_failed(&format!("{err:#}"), kind, guidance);
    if hi_agent::ui::error_counts_as_model_issue(&err) {
        app.record_model_issue();
    }

    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines.iter().any(|line| line.contains("request")),
        "transcript: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("degraded in-session")),
        "transcript: {lines:?}"
    );
    assert_eq!(app.model_issues.get("pipe/auto-coder"), None);
}

#[test]
fn empty_tool_result_is_visible() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "bash".into(),
        arguments: "{\"command\":\"true\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: String::new(),
    });
    let rendered: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        rendered.iter().any(|line| line.contains("(no output)")),
        "transcript: {rendered:?}"
    );
}

#[test]
fn explore_tools_collapse_header_and_line_count_into_one_line() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"src/main.rs\"}".into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines.iter().any(|l| l.contains("Reading src/main.rs")),
        "live read header before result: {lines:?}"
    );
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "a\nb\nc\n".into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines
            .iter()
            .any(|l| l.contains("Read src/main.rs · 3 lines")),
        "collapsed read line: {lines:?}"
    );
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.contains("Read src/main.rs"))
            .count(),
        1,
        "exactly one read header line: {lines:?}"
    );

    // grep folds into the same verb group (mixed exploration).
    app.apply(UiEvent::ToolCall {
        name: "grep".into(),
        arguments: "{\"pattern\":\"foo\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "grep".into(),
        result: String::new(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines
            .iter()
            .any(|l| l.contains("Read 1 file, Searched 1 pattern")),
        "mixed verb group: {lines:?}"
    );
}

#[test]
fn idle_bash_output_polls_collapse_into_one_updating_line() {
    let mut app = test_app("openai", "gpt-4o");
    let idle = "[sh_1: still running — no new output]";

    app.apply(UiEvent::ToolCall {
        name: "bash_output".into(),
        arguments: "{\"id\":\"sh_1\"}".into(),
    });
    // Live `Run` row as soon as the poll starts — never a raw tool dump.
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines.iter().any(|l| l.contains("Run sh_1")),
        "live Run row before result: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.contains("◆ bash_output") || l.contains("bash_output")),
        "no deferred header before result: {lines:?}"
    );

    app.apply(UiEvent::ToolResult {
        name: "bash_output".into(),
        result: idle.into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines.iter().any(|l| l.contains("Run sh_1")),
        "first idle poll: {lines:?}"
    );
    assert_eq!(
        lines.iter().filter(|l| l.contains("Run sh_1")).count(),
        1,
        "exactly one shell poll line: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.contains("no new output") || l.contains("still running")),
        "idle status body must not dump into the transcript: {lines:?}"
    );

    app.apply(UiEvent::ToolCall {
        name: "bash_output".into(),
        arguments: "{\"id\":\"sh_1\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash_output".into(),
        result: idle.into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "bash_output".into(),
        arguments: "{\"id\":\"sh_1\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash_output".into(),
        result: idle.into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert_eq!(
        lines.iter().filter(|l| l.contains("Run sh_1")).count(),
        1,
        "still exactly one shell poll line after three polls: {lines:?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("polled")),
        "poll count stays off the Run header: {lines:?}"
    );

    // Fresh output ends the collapse and shows a normal header + body.
    app.apply(UiEvent::ToolCall {
        name: "bash_output".into(),
        arguments: "{\"id\":\"sh_1\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash_output".into(),
        result: "[sh_1: still running]\n== hi-ai ==\n".into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        lines.iter().any(|l| l.contains("== hi-ai ==")),
        "fresh output is shown: {lines:?}"
    );
}

#[test]
fn live_run_streams_stdout_into_the_transcript() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "bash".into(),
        arguments: "{\"command\":\"cargo test\"}".into(),
    });
    app.apply(UiEvent::ToolStream {
        name: "bash".into(),
        line: "   Compiling hi v0.3.1".into(),
    });
    app.apply(UiEvent::ToolStream {
        name: "bash".into(),
        line: "    Finished `test` profile".into(),
    });
    let flat: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false, crate::Density::Comfortable))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(
        flat.iter().any(|l| l.contains("Run cargo")),
        "run header: {flat:?}"
    );
    assert!(
        flat.iter().any(|l| l.contains("Compiling hi")),
        "live stdout must be visible without Ctrl-O: {flat:?}"
    );
    assert!(
        flat.iter().any(|l| l.contains("Finished `test`")),
        "later stream lines stay on the same row: {flat:?}"
    );

    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "cargo test still running after 30s — continued as cargo-test_2.\n\
Use bash_output with id cargo-test_2 to read output."
            .into(),
    });
    let flat: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false, crate::Density::Comfortable))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(
        flat.iter().any(|l| l.contains("Compiling hi")),
        "auto-background keeps the live tail: {flat:?}"
    );
    let run_headers = flat.iter().filter(|l| l.contains("Run cargo")).count();
    assert_eq!(run_headers, 1, "one Run row after handoff: {flat:?}");

    app.apply(UiEvent::ToolCall {
        name: "bash_output".into(),
        arguments: "{\"id\":\"cargo-test_2\"}".into(),
    });
    app.apply(UiEvent::ToolStream {
        name: "bash_output".into(),
        line: "test result: ok. 455 passed".into(),
    });
    let flat: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false, crate::Density::Comfortable))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(
        flat.iter().any(|l| l.contains("test result: ok")),
        "bash_output streams onto the same Run row: {flat:?}"
    );
    let run_headers = flat.iter().filter(|l| l.contains("Run ")).count();
    assert_eq!(run_headers, 1, "still one Run row: {flat:?}");
}

#[test]
fn missing_background_poll_hides_model_recovery_instructions() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "bash_output".into(),
        arguments: r#"{"id":"git-status_1"}"#.into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash_output".into(),
        result: "Error: no background process `git-status_1` — no background processes are running at all. Do not call this again; continue the task with other tools.".into(),
    });

    let text = app.transcript_text();
    assert!(
        text.to_ascii_lowercase()
            .contains("background process git-status_1 unavailable")
    );
    assert!(!text.contains("no background process"));
    assert!(!text.contains("Do not call this again"));
}

#[test]
fn deepseek_wire_profile_status_is_not_user_visible() {
    let mut app = test_app("pipe", "deepseek-v4-flash");
    app.apply(UiEvent::Status {
        text: "compat: deepseek profile=gateway protocol=auto strict=false".into(),
    });

    assert!(!app.transcript_text().contains("compat: deepseek profile="));
}

#[test]
fn nested_deepseek_wire_profile_status_is_not_user_visible() {
    let mut app = test_app("pipe", "deepseek-v4-flash");
    app.apply(UiEvent::Status {
        text: "explore: compat: deepseek profile=gateway protocol=auto strict=false".into(),
    });

    assert!(!app.transcript_text().contains("compat: deepseek profile="));
}

#[test]
fn internal_steering_statuses_are_humanized_in_the_transcript() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Status {
        text: "turn stopped incomplete · repeat_no_op_bash".into(),
    });
    app.apply(UiEvent::Status {
        text: "⚠ the model kept emitting invalid tool turns — ending the turn; /retry or continue to resume".into(),
    });

    let text = app.transcript_text();
    assert!(text.contains("unfinished work"));
    assert!(text.contains("tool calls were invalid"));
    assert!(!text.contains("repeat_no_op_bash"));
    assert!(!text.contains("the model kept"));
    assert!(!text.contains("/retry"));
}

#[test]
fn model_only_background_instructions_are_hidden_from_tool_output() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "bash".into(),
        arguments: r#"{"command":"cargo test"}"#.into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "Started cargo test (sh_1). Use bash_output with id sh_1 for progress; bash_kill with id sh_1 to stop.".into(),
    });

    let text = app.transcript_text();
    assert!(text.contains("Started cargo test"));
    assert!(!text.contains("Use bash_output"));
    assert!(!text.contains("bash_kill"));
}

#[test]
fn consecutive_same_tool_explore_results_merge_into_one_line() {
    let mut app = test_app("openai", "gpt-4o");
    // Three reads in a row should collapse to one summary line.
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"a.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "a\nb\n".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"b.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "c\nd\ne\n".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"c.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "f\n".into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    // Exactly one read line, summarizing all three.
    assert_eq!(
        lines.iter().filter(|l| l.contains("Read ")).count(),
        1,
        "one merged read line: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("Read 3 files")),
        "merged summary: {lines:?}"
    );

    // A non-explore tool between reads breaks the run.
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"a.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "ok".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"d.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "x\ny\n".into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    // Now two read lines: the merged 3-file run and a fresh single read.
    assert_eq!(
        lines.iter().filter(|l| l.contains("Read ")).count(),
        2,
        "run broken by edit: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("Read d.rs · 2 lines")),
        "fresh read after break: {lines:?}"
    );
}

#[test]
fn explore_tools_group_across_assistant_narration() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "list".into(),
        arguments: "{\"path\":\".\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "list".into(),
        result: "a\nb\n".into(),
    });
    app.apply(UiEvent::Text {
        text: "Now let me look at the turn loop.\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    app.apply(UiEvent::Reasoning {
        text: "plan the next read".into(),
    });
    app.apply(UiEvent::Text {
        text: "Let me read the CLI entry point.\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"crates/hi-cli/src/main.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "fn main() {}\n".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "repo_map".into(),
        arguments: "{\"task\":\"overview\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "repo_map".into(),
        result: "map\n".into(),
    });
    let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.contains("Read ") || l.contains("Listed ") || l.contains("Repo_map"))
            .count(),
        1,
        "one explore row across narration: {lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("Listed 2 dirs") && l.contains("Read 1 file")),
        "mixed explore summary: {lines:?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("Repo_map")),
        "repo_map must fold into the explore row: {lines:?}"
    );
}

#[test]
fn collapsed_zero_second_thought_is_hidden() {
    let entry = TranscriptEntry::Reasoning {
        text: "instant".into(),
        elapsed: Duration::from_secs(0),
    };
    assert!(
        entry.flatten(false, false, Density::Comfortable).is_empty(),
        "collapsed 0s thought must not take a row"
    );
    assert!(
        !entry.flatten(true, false, Density::Comfortable).is_empty(),
        "Ctrl-T still reveals 0s thought"
    );
}

fn flatten_texts(app: &crate::App, show_reasoning: bool, show_tool: bool) -> Vec<String> {
    app.transcript
        .iter()
        .flat_map(|e| e.flatten(show_reasoning, show_tool, Density::Comfortable))
        .map(|l| crate::render::line_text(&l))
        .collect()
}

fn apply_read(app: &mut crate::App, path: &str, body: &str) {
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: format!(r#"{{"path":"{path}"}}"#),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: body.into(),
    });
}

#[test]
fn explore_burst_absorbs_thinking_and_steering() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Reasoning {
        text: "checking files".into(),
    });
    app.apply(UiEvent::Text {
        text: "Let me read a.rs\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    apply_read(&mut app, "a.rs", "one\n");
    apply_read(&mut app, "b.rs", "two\n");
    apply_read(&mut app, "c.rs", "three\n");

    let activities = app
        .transcript
        .iter()
        .filter(|e| matches!(e, TranscriptEntry::Activity(_)))
        .count();
    assert_eq!(
        activities,
        1,
        "one explore row: {:?}",
        app.transcript_text()
    );
    assert!(
        !app.transcript
            .iter()
            .any(|e| matches!(e, TranscriptEntry::Reasoning { .. })),
        "reasoning must be absorbed into the explore row"
    );

    let collapsed = flatten_texts(&app, false, false);
    assert!(
        collapsed.iter().any(|l| l.contains("Read 3 files")),
        "collapsed header: {collapsed:?}"
    );
    assert!(
        !collapsed
            .iter()
            .any(|l| l.contains("thought for") || l.contains("Let me read")),
        "thinking and steering stay folded: {collapsed:?}"
    );

    let ctrl_o = flatten_texts(&app, false, true);
    assert!(
        !ctrl_o
            .iter()
            .any(|l| l.contains("a.rs") || l.contains("b.rs")),
        "Ctrl-O must not dump explore paths: {ctrl_o:?}"
    );

    let with_t = flatten_texts(&app, true, false);
    assert!(
        with_t.iter().any(|l| l.contains("thought for")),
        "Ctrl-T shows thinking in the group: {with_t:?}"
    );
    assert!(
        with_t.iter().any(|l| l.contains("checking files")),
        "Ctrl-T shows thinking text: {with_t:?}"
    );
    assert!(
        !with_t.iter().any(|l| l.contains("Let me read")),
        "steering stays collapsed until Ctrl-B: {with_t:?}"
    );
}

#[test]
fn repeated_explore_steering_is_shown_once() {
    let mut app = test_app("openai", "gpt-4o");
    let steering = "Let me read the handle_mouse function to see the full event handling flow.\n";

    for path in ["a.rs", "b.rs", "c.rs"] {
        app.apply(UiEvent::Text {
            text: steering.into(),
        });
        app.apply(UiEvent::AssistantEnd);
        apply_read(&mut app, path, "fn main() {}\n");
    }

    let text = app.transcript_text();
    assert_eq!(
        text.matches("Let me read the handle_mouse function")
            .count(),
        1,
        "replayed explore preambles should be deduplicated: {text}"
    );
    assert!(
        text.contains("Read 3 files"),
        "explore row remains merged: {text}"
    );
}

#[test]
fn repeated_assistant_narration_is_printed_once_across_rounds() {
    let mut app = test_app("openai", "gpt-4o");
    let preamble = "I need to verify the ToolOutcome struct and caller sites before declaring this clean. Let me do that now.";
    let repeated = "Let me check the ToolOutcome struct and find callers of the changed functions.";

    app.apply(UiEvent::Text {
        text: format!("{preamble}\n"),
    });
    app.apply(UiEvent::AssistantEnd);
    for _ in 0..10 {
        app.apply(UiEvent::Text {
            text: format!("{repeated}\n"),
        });
        app.apply(UiEvent::AssistantEnd);
    }

    let text = app.transcript_text();
    assert_eq!(
        text.matches(repeated).count(),
        1,
        "identical narration must not be printed once per model round: {text}"
    );
    assert!(
        text.contains(preamble),
        "the distinct preamble remains: {text}"
    );
}

#[test]
fn repeated_generic_completion_ack_is_shown_once_per_turn() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);

    for _ in 0..10 {
        app.apply(UiEvent::Text {
            text: "Completed the requested action.\n".into(),
        });
        app.apply(UiEvent::AssistantEnd);
    }

    assert_eq!(
        app.transcript_text()
            .matches("Completed the requested action.")
            .count(),
        1,
        "generic completion acknowledgements should not repeat per retry round: {}",
        app.transcript_text()
    );

    // A new user turn may legitimately produce its own acknowledgement.
    app.set_working(false);
    app.set_working(true);
    app.apply(UiEvent::Text {
        text: "Completed the requested action.\n".into(),
    });
    assert_eq!(
        app.transcript_text()
            .matches("Completed the requested action.")
            .count(),
        2
    );

    // Do not hide a more informative completion sentence merely because it
    // starts with the same word.
    app.apply(UiEvent::Text {
        text: "Completed the requested action for all five files.\n".into(),
    });
    assert_eq!(
        app.transcript_text()
            .matches("Completed the requested action for all five files.")
            .count(),
        1
    );
}

#[test]
fn live_explore_steering_never_prints_as_a_separate_line() {
    let mut app = test_app("openai", "gpt-4o");
    let steering = "Let me read the handle_mouse function to see the full event handling flow.\n";

    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: r#"{"path":"src/main.rs"}"#.into(),
    });
    app.apply(UiEvent::Text {
        text: steering.into(),
    });

    assert_eq!(
        app.transcript
            .iter()
            .filter(|entry| matches!(entry, TranscriptEntry::AssistantMessage { .. }))
            .count(),
        0,
        "live steering should be owned by the explore row: {:?}",
        app.transcript_text()
    );
    let collapsed = flatten_texts(&app, false, false);
    assert!(
        !collapsed.iter().any(|line| line.contains("Let me read")),
        "collapsed feed should not print steering: {collapsed:?}"
    );
}

#[test]
fn repeated_steering_after_explore_result_is_folded_once() {
    let mut app = test_app("openai", "gpt-4o");
    apply_read(&mut app, "src/main.rs", "fn main() {}\n");
    let steering = "Let me check the ToolOutcome struct and find callers of the changed functions.";

    for _ in 0..10 {
        app.apply(UiEvent::Text {
            text: format!("{steering}\n"),
        });
        app.apply(UiEvent::AssistantEnd);
    }

    let text = app.transcript_text();
    assert_eq!(
        text.matches(steering).count(),
        1,
        "repeated post-tool steering should have one retained copy: {text}"
    );
    let collapsed = flatten_texts(&app, false, false);
    assert!(
        !collapsed.iter().any(|line| line.contains(steering)),
        "collapsed explore output should hide steering: {collapsed:?}"
    );
}

#[test]
fn substantial_answer_is_not_stolen_into_explore_group() {
    let mut app = test_app("openai", "gpt-4o");
    apply_read(&mut app, "a.rs", "one\n");
    app.apply(UiEvent::Text {
        text: "# Findings\nThe CLI entry is in main.rs.\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    apply_read(&mut app, "b.rs", "two\n");

    let activities = app
        .transcript
        .iter()
        .filter(|e| matches!(e, TranscriptEntry::Activity(_)))
        .count();
    assert_eq!(
        activities,
        2,
        "answer splits explore bursts: {:?}",
        app.transcript_text()
    );
    let texts: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("Findings")),
        "heading stays as the answer: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("CLI entry")),
        "paragraph stays as the answer: {texts:?}"
    );
}

#[test]
fn consecutive_edits_to_the_same_path_coalesce() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: r#"{"path":"src/a.rs"}"#.into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "\x1b[1m2 additions, 1 deletions\x1b[0m\n  10 + foo\n".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: r#"{"path":"src/a.rs"}"#.into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "\x1b[1m3 additions, 2 deletions\x1b[0m\n  11 + bar\n".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: r#"{"path":"src/b.rs"}"#.into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "\x1b[1m1 additions, 0 deletions\x1b[0m\n  1 + baz\n".into(),
    });

    let edits: Vec<String> = app
        .transcript
        .iter()
        .filter(|e| matches!(e, TranscriptEntry::Activity(_)))
        .map(TranscriptEntry::text)
        .collect();
    assert_eq!(edits.len(), 2, "same-path edits merge: {edits:?}");
    assert!(
        edits.iter().any(|t| t.contains("Edit a.rs +5/-3")),
        "coalesced diffstat: {edits:?}"
    );
    assert!(
        edits.iter().any(|t| t.contains("Edit b.rs +1")),
        "different path stays a second row: {edits:?}"
    );

    let collapsed = flatten_texts(&app, false, false);
    assert!(
        !collapsed
            .iter()
            .any(|l| l.contains("+ foo") || l.contains("+ bar")),
        "hunks stay collapsed: {collapsed:?}"
    );
    let expanded = flatten_texts(&app, false, true);
    assert!(
        expanded.iter().any(|l| l.contains("foo")) && expanded.iter().any(|l| l.contains("bar")),
        "Ctrl-O still shows hunks: {expanded:?}"
    );
}

#[test]
fn assistant_end_collapses_streamed_reply_into_one_message() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Text {
        text: "hello\nworld".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    let messages: Vec<_> = app
        .transcript
        .iter()
        .filter(|e| matches!(e, TranscriptEntry::AssistantMessage { .. }))
        .collect();
    assert_eq!(
        messages.len(),
        1,
        "{:?}",
        app.transcript
            .iter()
            .map(TranscriptEntry::text)
            .collect::<Vec<_>>()
    );
    assert_eq!(messages[0].text(), "hello\nworld");
}

#[test]
fn page_flip_pins_sent_prompt_when_working() {
    let mut app = test_app("openai", "gpt-4o");
    app.following = true;
    for i in 0..20 {
        app.transcript
            .push(TranscriptEntry::Assistant(Line::raw(format!("pad {i}"))));
    }
    app.working = true;
    app.push_user_prompt(Line::raw("new task"));
    assert!(app.page_flip_on_send);
    let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    for i in 0..30 {
        app.transcript
            .push(TranscriptEntry::Assistant(Line::raw(format!("later {i}"))));
    }
    terminal.draw(|frame| app.render(frame)).unwrap();
    assert!(
        !app.following,
        "page-flip stays unpinned so the reply grows under the prompt"
    );
    let screen = dump(&terminal);
    assert!(
        screen.contains("new task"),
        "sent prompt must stay visible after page-flip:\n{screen}"
    );
    assert!(
        !screen.contains("later 29"),
        "must not jump to the tail while page-flipped:\n{screen}"
    );
}

#[test]
fn markdown_heading_and_list_gain_blank_lines() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Text {
        text: "intro\n# Title\nbody\n- item".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    let texts = flatten_texts(&app, false, false);
    assert_eq!(
        texts,
        vec![
            "intro".to_string(),
            String::new(),
            "Title".to_string(),
            String::new(),
            "body".to_string(),
            String::new(),
            "• item".to_string(),
        ],
        "blank before/after H1 and before the list: {texts:?}"
    );
}

#[test]
fn edit_activity_row_shows_colored_diffstat() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"src/chrome.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "\x1b[1m12 additions, 3 deletions\x1b[0m\n  10 + new\n  11 - old\n".into(),
    });
    let text = app
        .transcript
        .iter()
        .map(TranscriptEntry::text)
        .collect::<Vec<_>>();
    assert!(
        text.iter().any(|l| l.contains("Edit chrome.rs +12/-3")),
        "collapsed edit identity: {text:?}"
    );
    let collapsed: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false, crate::Density::Comfortable))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(
        collapsed.len() > 1,
        "edit keeps a compact colored preview: {collapsed:?}"
    );
    assert!(
        collapsed.iter().any(|line| line.contains("new")),
        "collapsed edit preview shows changed content: {collapsed:?}"
    );
    let collapsed_styles: Vec<Option<Color>> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false, crate::Density::Comfortable))
        .flat_map(|line| line.spans.into_iter().map(|span| span.style.fg))
        .collect();
    assert!(
        collapsed_styles.contains(&Some(crate::theme::theme().diff_add)),
        "collapsed edit preview colors additions: {collapsed_styles:?}"
    );
    assert!(
        collapsed_styles.contains(&Some(crate::theme::theme().diff_del)),
        "collapsed edit preview colors deletions: {collapsed_styles:?}"
    );
    let compact: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false, crate::Density::Compact))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert_eq!(
        compact.len(),
        1,
        "compact density keeps edits to their header: {compact:?}"
    );
    let expanded: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, true, crate::Density::Comfortable))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(
        expanded.iter().any(|l| l.contains("new")),
        "expanded hunks: {expanded:?}"
    );
}

#[test]
fn renders_fetching_spinner() {
    let mut app = test_app("pipenetwork", "ipop/coder-balanced");
    app.fetching = Some(Instant::now());
    let mut term = Terminal::new(TestBackend::new(60, 10)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("fetching models from pipenetwork"),
        "fetch spinner: {screen}"
    );
    assert!(screen.contains("Esc to cancel"), "cancel hint: {screen}");
}

#[test]
fn renders_model_picker() {
    let mut app = test_app("openai", "openai/gpt-4o");
    app.picker = Some(ModelPicker::new(
        vec!["anthropic/claude-sonnet-4".into(), "openai/gpt-4o".into()],
        "openai/gpt-4o",
        HashMap::new(),
        &HashMap::new(),
    ));
    let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("select a model"), "title: {screen}");
    assert!(screen.contains("filter:"), "filter line: {screen}");
    assert!(screen.contains("claude-sonnet-4"), "lists models: {screen}");
    assert!(screen.contains("▶"), "highlights a selection: {screen}");
    // The active model is marked and pre-selected.
    assert!(
        screen.contains("(current)"),
        "marks current model: {screen}"
    );
}

#[test]
fn picker_hides_health_tag() {
    let mut app = test_app("pipenetwork", "ipop/coder-balanced");
    let tags = HashMap::from([("claude-sonnet-4.6".to_string(), "degraded".to_string())]);
    app.picker = Some(ModelPicker::new(
        vec!["claude-sonnet-4.6".into(), "ipop/coder-balanced".into()],
        "ipop/coder-balanced",
        tags,
        &HashMap::new(),
    ));
    let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        !screen.contains("[degraded]"),
        "health tag should not be shown: {screen}"
    );
}

#[test]
fn renders_multiline_input() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.insert_str("first\nsecond\nthird");
    let mut term = Terminal::new(TestBackend::new(40, 14)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("❯ first"),
        "first line with prompt: {screen}"
    );
    assert!(screen.contains("second"), "second line: {screen}");
    assert!(screen.contains("third"), "third line: {screen}");
}

#[test]
fn modified_enter_and_backslash_insert_newline_instead_of_submitting() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("line one");
    let alt_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
    assert_eq!(app.edit_key(&alt_enter), None, "alt+enter does not submit");
    assert_eq!(app.input.text(), "line one\n");

    app.input.set("line two");
    let shift_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT);
    assert_eq!(
        app.edit_key(&shift_enter),
        None,
        "shift+enter does not submit"
    );
    assert_eq!(app.input.text(), "line two\n");

    // Trailing backslash + Enter continues the line (universal fallback).
    app.input.set("a\\");
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(app.edit_key(&enter), None, "backslash continues");
    assert_eq!(app.input.text(), "a\n");

    // A normal Enter still submits.
    app.input.set("go");
    assert_eq!(app.edit_key(&enter).as_deref(), Some("go"));
}

#[test]
fn failed_turn_shows_reason_and_keeps_error() {
    let mut app = test_app("openai", "gpt-4o");
    app.note_turn_failed(
        "API error 401: invalid or expired session",
        "auth",
        "check your API key",
    );
    // record_model_issue runs next in the real flow; it must NOT clobber the
    // real error with a reliability-count message.
    app.record_model_issue();
    assert_eq!(
        app.last_error.as_deref(),
        Some("API error 401: invalid or expired session"),
        "the real error is preserved for /status and /log"
    );
    // The bottom bar shows the reason inline, not a bare "failed".
    let mut term = Terminal::new(TestBackend::new(80, 8)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("last: failed — API error 401"),
        "reason inline: {screen}"
    );
    assert!(screen.contains("/retry"), "recovery hint: {screen}");
}

#[test]
fn backend_wait_notice_does_not_mark_model_degraded() {
    let mut app = test_app("pipenetwork", "ipop/coder-balanced");
    app.note_backend_waiting(Duration::from_secs(181), Duration::from_secs(180));

    assert_eq!(app.model_issues.get("ipop/coder-balanced"), None);
    assert_eq!(app.last_error, None);
    let mut term = Terminal::new(TestBackend::new(100, 8)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Still thinking. Ctrl-C cancels; keep waiting to continue."),
        "soft wait notice shown: {screen}"
    );
    assert!(
        !screen.contains("degraded in-session"),
        "soft wait notice should not surface model health: {screen}"
    );
}

#[test]
fn watchdog_timeout_default_is_longer_than_client_warning_window() {
    assert_eq!(
        watchdog_stuck_timeout_from_value(None),
        Duration::from_secs(180)
    );
    assert_eq!(
        watchdog_stuck_timeout_from_value(Some("5")),
        Duration::from_secs(30)
    );
    assert_eq!(
        watchdog_stuck_timeout_from_value(Some("9999")),
        Duration::from_secs(1_800)
    );
}

#[test]
fn completion_opens_filters_and_closes() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("/");
    app.sync_completion();
    assert_eq!(
        app.completion_items().len(),
        hi_agent::command::COMMANDS.len() + 3,
        "bare slash lists every agent command and tutorial alias"
    );
    let tutorial_labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|item| item.label.clone())
        .filter(|label| matches!(label.as_str(), "/tutorial" | "/tour" | "/onboarding"))
        .collect();
    assert_eq!(tutorial_labels.len(), 3);
    app.input.set("/co");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert!(
        labels.contains(&"/copy".to_string()) && labels.contains(&"/compact".to_string()),
        "got {labels:?}"
    );
    assert!(labels.iter().all(|n| n.starts_with("/co")));
    // A space after a command that takes no argument closes the menu.
    app.input.set("/diff ");
    app.sync_completion();
    assert!(app.completion.is_none());
}

#[test]
fn history_recall_of_slash_command_keeps_completion_closed() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.history = vec!["ask first".into(), "/help".into(), "ask last".into()];
    let up = KeyEvent::new(KeyCode::Up, KeyModifiers::NONE);

    assert_eq!(app.edit_key(&up), None);
    app.sync_completion_after_edit_key(&up, false);
    assert_eq!(app.input.text(), "ask last");
    assert!(app.completion.is_none());

    assert_eq!(app.edit_key(&up), None);
    app.sync_completion_after_edit_key(&up, false);
    assert_eq!(app.input.text(), "/help");
    assert!(
        app.completion.is_none(),
        "history recall must not open slash completion"
    );

    assert_eq!(app.edit_key(&up), None);
    app.sync_completion_after_edit_key(&up, false);
    assert_eq!(app.input.text(), "ask first");
    assert!(app.completion.is_none());
}

#[test]
fn history_search_recall_of_slash_command_keeps_completion_closed() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.history = vec!["ask first".into(), "/help".into()];
    let mut search = HistorySearch::default();
    search.refilter(&app.input.history);
    app.mode = crate::mode::UiMode::HistorySearch(search);

    let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
    let history_search_was_active = app.mode.is_history_search();
    assert_eq!(app.edit_key(&esc), None);
    app.sync_completion_after_edit_key(&esc, history_search_was_active);

    assert_eq!(app.input.text(), "/help");
    assert!(
        app.completion.is_none(),
        "loading a slash command from Ctrl-R should leave arrows for history"
    );

    app.sync_completion();
    assert!(
        app.completion.is_some(),
        "normal slash completion remains available outside history recall"
    );
}

#[test]
fn completion_offers_verify_and_goal_keywords() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("/verify ");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert_eq!(labels, vec!["off"], "verify offers its disable keyword");
    app.input.set("/goal cl");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert_eq!(labels, vec!["clear"], "goal offers its clear keyword");
    assert_eq!(app.accept_completion(true).as_deref(), Some("/goal clear"));
}

#[test]
fn completion_offers_live_model_ids() {
    let mut app = test_app("openai", "gpt-4o");
    app.model_ids = vec!["gpt-4o".into(), "gpt-4o-mini".into(), "claude-opus".into()];
    app.input.set("/model gp");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert_eq!(
        labels,
        vec!["gpt-4o", "gpt-4o-mini"],
        "filters the catalog by prefix"
    );
    // Accepting a row runs the full command.
    app.completion.as_mut().unwrap().selected = 1;
    assert_eq!(
        app.accept_completion(true).as_deref(),
        Some("/model gpt-4o-mini")
    );

    // With no catalog loaded, there's no inline menu — the picker still
    // handles `/model` (so the feature degrades, it doesn't break).
    let mut bare = test_app("openai", "gpt-4o");
    bare.input.set("/model gp");
    bare.sync_completion();
    assert!(bare.completion.is_none());
}

#[test]
fn sessions_completion_offers_subcommands_then_live_ids() {
    let mut app = test_app("openai", "gpt-4o");
    app.session_lister = Some(Box::new(|| {
        vec![
            LocalSessionInfo {
                id: "1783895144561".into(),
                title: "portal work".into(),
                age: "2m".into(),
                lines: 12,
            },
            LocalSessionInfo {
                id: "1783894593132".into(),
                title: "other work".into(),
                age: "8m".into(),
                lines: 4,
            },
        ]
    }));

    app.input.set("/sessions sw");
    app.sync_completion();
    assert_eq!(app.completion_items()[0].label, "switch");
    assert_eq!(app.accept_completion(true), None);
    assert_eq!(app.input.text(), "/sessions switch ");

    app.sync_completion();
    assert_eq!(app.completion_items().len(), 2);
    assert_eq!(
        app.accept_completion(true).as_deref(),
        Some("/sessions switch 1783895144561")
    );

    app.input.set("/sessions rename 1783894");
    app.sync_completion();
    assert_eq!(app.accept_completion(true), None);
    assert_eq!(app.input.text(), "/sessions rename 1783894593132 ");
}

#[test]
fn session_completion_does_not_rescan_files_for_each_prefix_or_render() {
    let mut app = test_app("openai", "gpt-4o");
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = calls.clone();
    app.session_lister = Some(Box::new(move || {
        observed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        vec![LocalSessionInfo {
            id: "1783895144561".into(),
            title: "portal work".into(),
            age: "2m".into(),
            lines: 12,
        }]
    }));

    app.input.set("/sessions switch ");
    app.sync_completion();
    assert_eq!(app.completion_items().len(), 1);
    app.input.set("/sessions switch 178");
    app.sync_completion();
    for _ in 0..5 {
        assert_eq!(app.completion_items().len(), 1);
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn completion_offers_then_fills_compact_kinds() {
    let mut app = test_app("openai", "gpt-4o");
    // The space that used to kill the menu now offers the kinds.
    app.input.set("/compact ");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert_eq!(labels, vec!["hybrid", "full", "elide"], "offers every kind");
    // Typing narrows by prefix.
    app.input.set("/compact e");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert_eq!(labels, vec!["elide"]);
    // Accepting a kind fills the whole command and runs it on Enter.
    assert_eq!(
        app.accept_completion(true).as_deref(),
        Some("/compact elide")
    );
    assert!(app.completion.is_none(), "menu closes after accept");
}

#[test]
fn completing_compact_name_opens_its_kind_menu() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("/compact");
    app.sync_completion();
    // Tab accepts the command name, leaving `/compact `…
    app.accept_completion(false);
    assert_eq!(app.input.text(), "/compact ");
    // …and the re-sync the Tab handler performs opens the kind menu.
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert!(labels.contains(&"hybrid".to_string()), "got {labels:?}");
}

#[test]
fn completion_navigation_and_accept() {
    let mut app = test_app("openai", "gpt-4o");
    // No-arg command: Enter accepts and submits immediately.
    app.input.set("/und");
    app.sync_completion();
    let line = app.accept_completion(true);
    assert_eq!(line.as_deref(), Some("/undo"));
    assert!(app.completion.is_none(), "menu closes after accept");

    // Arg-taking command: accept leaves a trailing space, does not submit.
    app.input.set("/mod");
    app.sync_completion();
    assert_eq!(
        app.accept_completion(true),
        None,
        "arg command waits for input"
    );
    assert_eq!(app.input.text(), "/model ");

    // Tab never submits, even for a no-arg command.
    app.input.set("/dif");
    app.sync_completion();
    assert_eq!(app.accept_completion(false), None);
    assert_eq!(app.input.text(), "/diff");
}

#[test]
fn completion_move_clamps() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("/co"); // [commit, compact, context, copy]
    app.sync_completion();
    let last = app.completion_items().len().saturating_sub(1);
    app.completion_move(-1); // already at 0, stays
    assert_eq!(app.completion.as_ref().unwrap().selected, 0);
    app.completion_move(1);
    assert_eq!(app.completion.as_ref().unwrap().selected, 1);
    // Move past the end to verify clamping.
    for _ in 0..last + 1 {
        app.completion_move(1);
    }
    assert_eq!(app.completion.as_ref().unwrap().selected, last);
}

#[test]
fn renders_completion_menu() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("/");
    app.sync_completion();
    let mut term = Terminal::new(TestBackend::new(72, 20)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("/help"), "lists help: {screen}");
    assert!(screen.contains("/model"), "lists model: {screen}");
    assert!(screen.contains("▶"), "highlights a row: {screen}");
}

/// A fresh session does not dump a keybinding essay into the transcript.
/// Chrome already has cwd, model, and shortcuts; the canvas shows the
/// figlet wordmark until the first turn.
#[test]
fn empty_session_landing_is_quiet() {
    let mut app = test_app("openai", "gpt-4o");
    assert!(
        app.transcript.is_empty(),
        "fresh session must not seed the transcript"
    );

    let mut term = Terminal::new(TestBackend::new(80, 20)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);

    assert!(
        screen.contains("|_| |_|_|"),
        "wide canvas shows the wordmark: {screen}"
    );
    assert!(
        !screen.contains("Alt-Enter"),
        "no keybinding essay: {screen}"
    );
    assert!(
        !screen.contains("ephemeral execution") && !screen.contains("durable execution"),
        "no execution lecture: {screen}"
    );
    assert!(
        !screen.contains("type a task"),
        "empty canvas is the wordmark, not a prompt lecture: {screen}"
    );
}

#[test]
fn top_status_warning_stays_out_of_composer() {
    let mut app = test_app("pipe", "glm-5.2");
    app.apply(UiEvent::TopStatus {
        text:
            "Outcome unavailable (RSI worker heartbeat is not ready); falling back to local chat."
                .into(),
    });
    app.input.set("next");

    let mut term = Terminal::new(TestBackend::new(100, 20)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    let rows: Vec<&str> = screen.lines().collect();

    assert!(
        rows[0].contains("Outcome unavailable"),
        "provider warning belongs in the top status bar: {screen}"
    );
    assert!(
        !app.transcript_text().contains("Outcome unavailable"),
        "provider warning must not become transcript content"
    );
    let composer_rows = rows
        .iter()
        .filter(|row| row.contains('╭') || row.contains('❯') || row.contains('╰'))
        .copied()
        .collect::<Vec<_>>();
    assert!(
        composer_rows
            .iter()
            .all(|row| !row.contains("Outcome unavailable")),
        "provider warning must not be painted into the composer: {composer_rows:?}"
    );
}

#[test]
fn checkpoint_warning_is_pinned_without_transcript_copy() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::CheckpointWarning {
        text: "⚠ could not seal this turn's undo record: git unavailable".into(),
    });

    let mut term = Terminal::new(TestBackend::new(100, 20)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    let rows: Vec<&str> = screen.lines().collect();

    assert!(
        rows[0].contains("could not seal"),
        "checkpoint warning belongs in the top status bar: {screen}"
    );
    assert_eq!(
        rows[0].matches('⚠').count(),
        1,
        "top status should render one warning marker: {screen}"
    );
    assert!(
        screen.contains("undo warning — see the top bar"),
        "composer keeps only a compact warning affordance: {screen}"
    );
    assert!(
        !app.transcript_text().contains("could not seal"),
        "checkpoint warning should not be copied into the transcript"
    );
}

#[test]
fn changed_files_event_updates_one_chrome_summary_without_transcript_duplicate() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ChangedFiles {
        files: vec!["src/a.rs".into(), "src/b.rs".into()],
    });

    assert_eq!(app.transcript_text(), "");
    assert_eq!(app.last_changed_files, ["src/a.rs", "src/b.rs"]);
    assert_eq!(app.session_changed_files, ["src/a.rs", "src/b.rs"]);

    let mut term = Terminal::new(TestBackend::new(80, 20)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert_eq!(screen.matches("changed: src/a.rs, src/b.rs").count(), 1);

    let hit = app.changed_files_rect;
    app.handle_mouse(crossterm::event::MouseEvent {
        kind: crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left),
        column: hit.x.saturating_add(1),
        row: hit.y,
        modifiers: crossterm::event::KeyModifiers::NONE,
    });
    assert!(
        app.mode.is_review(),
        "changed-files row opens filtered review"
    );
}

#[test]
fn normalized_steering_repeats_once_per_turn_but_resets_for_next_turn() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    for text in [
        "Let me check the files.\n",
        "  Let   me check the files.  \n",
    ] {
        app.apply(UiEvent::Text { text: text.into() });
        app.apply(UiEvent::AssistantEnd);
    }
    assert_eq!(app.transcript_text().matches("Let me check").count(), 1);

    app.set_working(false);
    app.set_working(true);
    app.apply(UiEvent::Text {
        text: "Let me check the files.\n".into(),
    });
    assert_eq!(app.transcript_text().matches("Let me check").count(), 2);
}

#[test]
fn substantive_answers_are_not_deduplicated() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    for _ in 0..2 {
        app.apply(UiEvent::Text {
            text: "The result is stable.\n".into(),
        });
        app.apply(UiEvent::AssistantEnd);
    }

    assert_eq!(
        app.transcript_text()
            .matches("The result is stable.")
            .count(),
        2,
        "substantive repeated answers must remain visible"
    );
}

#[test]
fn working_status_replaces_one_chrome_notice_instead_of_repeating() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Status {
        text: "still working — first step".into(),
    });
    app.apply(UiEvent::Status {
        text: "still working — second step".into(),
    });

    assert!(app.transcript.is_empty(), "working status is chrome-only");
    assert_eq!(
        app.working_status.as_deref(),
        Some("still working — second step")
    );

    let mut term = Terminal::new(TestBackend::new(100, 20)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    assert!(
        dump(&term).contains("still working — second step"),
        "latest working status should be visible in the top chrome"
    );
}

#[test]
fn uievent_serializes_and_deserializes_roundtrip() {
    use crate::event::UiEvent;
    use hi_agent::{PlanStatus, PlanStep};

    // Every variant must round-trip through serde JSON.
    let cases = vec![
        UiEvent::Text {
            text: "hello".to_string(),
        },
        UiEvent::Reasoning {
            text: "thinking...".to_string(),
        },
        UiEvent::AssistantEnd,
        UiEvent::ToolStarted {
            name: "bash".to_string(),
            arguments: r#"{"command":"ls"}"#.to_string(),
        },
        UiEvent::ToolCall {
            name: "edit".to_string(),
            arguments: r#"{"path":"a.rs"}"#.to_string(),
        },
        UiEvent::ToolResult {
            name: "bash".to_string(),
            result: "ok".to_string(),
        },
        UiEvent::ToolStream {
            name: "bash".to_string(),
            line: "compiling...".to_string(),
        },
        UiEvent::Status {
            text: "running".to_string(),
        },
        UiEvent::TopStatus {
            text: "Outcome unavailable; falling back to local chat".to_string(),
        },
        UiEvent::Plan {
            steps: vec![
                PlanStep {
                    title: "step 1".to_string(),
                    status: PlanStatus::Done,
                },
                PlanStep {
                    title: "step 2".to_string(),
                    status: PlanStatus::Active,
                },
            ],
        },
        UiEvent::Usage {
            prompt: 100,
            generated: 50,
            ctx_used: 1000,
            ctx_window: Some(8000),
            estimated: false,
        },
        UiEvent::RateLimits { rate_limits: None },
        UiEvent::SubagentSpawned {
            id: "explore-1".to_string(),
            subagent_kind: "explore".to_string(),
            description: "crate boundaries".to_string(),
            background: false,
        },
        UiEvent::SubagentProgress {
            id: "explore-1".to_string(),
            activity: "Reading lib.rs".to_string(),
            line: Some("read lib.rs".to_string()),
        },
        UiEvent::SubagentFinished {
            id: "explore-1".to_string(),
            status: "completed".to_string(),
            elapsed_ms: 1200,
            summary: "found it".to_string(),
        },
        UiEvent::TurnEnd {
            summary: "[100 in · 50 out]".to_string(),
        },
        UiEvent::TurnError {
            error_kind: "rate_limit".to_string(),
            message: "too many requests".to_string(),
            guidance: "wait and retry".to_string(),
        },
        UiEvent::ChangedFiles {
            files: vec!["a.rs".to_string(), "b.rs".to_string()],
        },
    ];

    for original in &cases {
        let json = serde_json::to_string(original).unwrap();
        let decoded: UiEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(
            serde_json::to_string(&decoded).unwrap(),
            json,
            "round-trip mismatch for {json}"
        );
    }

    // Verify the tagged format: each event has a "kind" field.
    let text_json = serde_json::to_string(&UiEvent::Text {
        text: "hi".to_string(),
    })
    .unwrap();
    assert!(
        text_json.contains(r#""kind":"text""#),
        "text event should have kind tag: {text_json}"
    );
    assert!(
        text_json.contains(r#""text":"hi""#),
        "text event should have text field: {text_json}"
    );

    // Verify the TurnError uses error_kind (not kind, which conflicts with the tag).
    let error_json = serde_json::to_string(&UiEvent::TurnError {
        error_kind: "auth".to_string(),
        message: "bad key".to_string(),
        guidance: "check key".to_string(),
    })
    .unwrap();
    assert!(
        error_json.contains(r#""error_kind":"auth""#),
        "turn_error should use error_kind field: {error_json}"
    );
    assert!(
        !error_json.contains(r#""kind":"auth""#),
        "turn_error must not use kind for the error type (conflicts with tag): {error_json}"
    );
}

/// A visual smoke of the Phase-1 transcript grammar: prints the rendered screen
/// (run with `--nocapture`) and asserts the block-accent markers are present.
#[test]
fn phase1_visual_grammar_smoke() {
    let mut app = test_app("pipe", "glm-5.2");
    app.push(ratatui::text::Line::styled(
        "❯ port the parser to the new API",
        ratatui::style::Style::default().fg(crate::theme::theme().accent_user),
    ));
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: "{\"path\":\"src/parser.rs\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "read".into(),
        result: "a\nb\nc\n".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "bash".into(),
        arguments: "{\"command\":\"cargo test\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "running 3 tests\ntest result: ok".into(),
    });
    app.apply(UiEvent::Status {
        text: "🔍 skeptic approved — advancing".into(),
    });
    app.apply(UiEvent::Text {
        text: "Done. The parser now uses the new API.\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    app.apply(UiEvent::ChangedFiles {
        files: vec!["src/parser.rs".into()],
    });

    // Exercise the chrome: context chip + input.
    app.context_used = 42000;
    app.context_window = Some(128000);
    let mut term = Terminal::new(TestBackend::new(72, 22)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    println!("\n{screen}");

    assert!(screen.contains("❯ port the parser"), "user prompt band");
    assert!(
        screen.contains("Read src/parser.rs") || screen.contains("Read parser.rs"),
        "read activity row: {screen}"
    );
    assert!(
        screen.contains("Run cargo test") || screen.contains("Run cargo"),
        "run activity row: {screen}"
    );
    assert!(
        !screen.contains("skeptic approved"),
        "status chatter stays out of the activity list: {screen}"
    );
    assert!(
        screen.contains("changed: src/parser.rs"),
        "changed-files line"
    );
}

#[test]
fn long_tool_output_folds_to_preview_and_expands_on_ctrl_o() {
    let mut app = test_app("pipe", "glm-5.2");
    // 40 lines of bash output — well over the preview cap.
    let output = (0..40)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    app.apply(UiEvent::ToolCall {
        name: "bash".into(),
        arguments: "{\"command\":\"seq 40\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: output,
    });

    // Collapsed (default): Run header + first 2 / last 3, not the full dump.
    let collapsed: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false, crate::Density::Comfortable))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(
        collapsed.iter().any(|l| l.contains("Run seq")),
        "collapsed run header: {collapsed:?}"
    );
    assert!(
        collapsed.iter().any(|l| l.contains("… +")),
        "middle stdout is folded: {collapsed:?}"
    );
    assert!(
        !collapsed.iter().any(|l| l.contains("line 20")),
        "the middle is folded away when collapsed: {collapsed:?}"
    );

    // Expanded (Ctrl-O / show_tool_output): the full body, no footer.
    let expanded: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, true, crate::Density::Comfortable))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(
        expanded.iter().any(|l| l.contains("line 39")),
        "expanded shows the whole output"
    );
    assert!(
        !expanded.iter().any(|l| l.contains("Ctrl-O to expand")),
        "no fold footer when expanded"
    );

    // Short output is still a one-liner; the body is in copy/export text.
    let mut app2 = test_app("pipe", "glm-5.2");
    app2.apply(UiEvent::ToolResult {
        name: "bash".into(),
        result: "just one line".into(),
    });
    let short: Vec<String> = app2
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false, crate::Density::Comfortable))
        .map(|l| crate::render::line_text(&l))
        .collect();
    assert!(
        short.iter().any(|l| l.contains("Run bash")),
        "short run stays a header: {short:?}"
    );
    assert!(
        !short.iter().any(|l| l.contains("Ctrl-O")),
        "short output isn't a fold footer"
    );
    assert!(
        app2.transcript
            .iter()
            .any(|e| e.text().contains("just one line")),
        "copy/export keeps the short body"
    );

    // Full text (for /copy and /export) always has everything, regardless of fold.
    let full = app
        .transcript
        .iter()
        .map(TranscriptEntry::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(full.contains("line 39"), "copy/export keeps the full body");
}

#[test]
fn tool_output_body_carries_panel_background_when_theme_paints() {
    // Mutation-free: flatten reads the active theme; assert the body-line bg is
    // consistent with whatever palette is active (panel on truecolor, none on
    // ansi). This never touches the global mode, so it can't race other tests.
    let th = crate::theme::theme();
    let body: Vec<Line<'static>> = vec![Line::raw("a line of output")];
    let entry = TranscriptEntry::ToolOutput {
        body,
        expanded: false,
    };
    let flat = entry.flatten(false, true, crate::Density::Comfortable);
    let bg = flat[0].style.bg;
    if th.paints_backgrounds() {
        assert_eq!(
            bg,
            Some(th.panel),
            "truecolor themes sink the body into a panel"
        );
    } else {
        assert_eq!(
            bg, None,
            "ansi theme leaves the body background at terminal default"
        );
    }
}

#[test]
fn sticky_prompt_header_pins_when_scrolled_past() {
    let mut app = test_app("pipe", "glm-5.2");
    app.push_user_prompt(Line::styled(
        "❯ first question about the parser",
        Style::default().fg(crate::theme::theme().accent_user),
    ));
    // A long block of output so the prompt scrolls off the top.
    for i in 0..60 {
        app.push(Line::raw(format!("output line {i}")));
    }
    let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();

    // First render (following = bottom-pinned): NO sticky. Row 0 is the
    // status bar; row 1 is the first visible transcript line.
    term.draw(|f| app.render(f)).unwrap();
    let bottom_pinned = dump(&term);
    let first_content_row = bottom_pinned.lines().nth(1).unwrap_or("");
    assert!(
        !first_content_row.contains("first question"),
        "while following, the prompt is not pinned: {first_content_row:?}"
    );

    // Scroll up to the top: the prompt is now above the viewport → pinned.
    app.following = false;
    app.scroll = 0; // top of transcript
    // At scroll 0 the prompt IS visible (offset 0 == scroll), so it should NOT
    // pin. Now scroll down past it.
    app.scroll = 30;
    term.draw(|f| app.render(f)).unwrap();
    let scrolled = dump(&term);
    let top_content_row = scrolled.lines().nth(1).unwrap_or("");
    assert!(
        top_content_row.contains("first question"),
        "the governing prompt pins to the top when scrolled past: {top_content_row:?}"
    );
}

/// `/provider pipe` must complete the hosted preset. Completing only profile
/// names made pipenetwork unreachable after `/login pipenetwork` when no
/// profile was named that way.
#[test]
fn provider_completion_offers_the_pipenetwork_preset() {
    let app = test_app("xai", "grok-4.3");
    let labels = |prefix: &str| -> Vec<String> {
        app.provider_completion_items(prefix)
            .into_iter()
            .map(|item| item.label)
            .collect()
    };

    let all = labels("");
    assert!(
        all.contains(&"pipenetwork".into()),
        "empty /provider prefix must list the hosted preset: {all:?}"
    );
    assert!(
        all.contains(&"xai".into()),
        "other presets stay listed: {all:?}"
    );

    let pipe = labels("pipe");
    assert_eq!(
        pipe,
        vec!["pipenetwork".to_string()],
        "/provider pipe must select pipenetwork, not close the menu: {pipe:?}"
    );
    let item = app
        .provider_completion_items("pipe")
        .into_iter()
        .next()
        .expect("pipenetwork completion");
    assert_eq!(item.insert, "/provider pipenetwork");
    assert!(item.submit_on_enter);
}

/// A profile named after a provider still shadows the preset in completion,
/// matching picker resolution (profiles win on a name clash).
#[test]
fn provider_completion_does_not_duplicate_a_profile_named_pipenetwork() {
    let mut app = test_app("xai", "grok-4.3");
    app.profiles = vec![crate::ProfileInfo {
        name: "pipenetwork".into(),
        provider: "pipenetwork".into(),
        model: Some("ipop/coder-balanced".into()),
        base_url: None,
        managed_local_repo: None,
        managed_local_path: None,
    }];
    let pipe: Vec<String> = app
        .provider_completion_items("pipe")
        .into_iter()
        .map(|item| format!("{}|{}", item.label, item.help))
        .collect();
    assert_eq!(
        pipe.len(),
        1,
        "one pipenetwork row, not preset+profile: {pipe:?}"
    );
    assert!(
        pipe[0].contains("ipop/coder-balanced"),
        "the configured profile must win: {pipe:?}"
    );
}

/// `/provider xai` switches to a provider preset without creating a profile,
/// so the active name need not name one. Selecting a model then had nothing to
/// persist into and surfaced "couldn't save model to active profile: no profile
/// named 'xai'" over an otherwise successful switch.
#[test]
fn selecting_a_model_on_a_provider_preset_does_not_error_about_a_missing_profile() {
    let mut app = test_app("xai", "grok-4.3");
    // No profiles configured; the active name is a provider preset.
    app.active_profile = Some("xai".to_string());

    let saved = app
        .persist_active_profile_model("grok-4.5")
        .expect("a preset with no profile must not be an error");
    assert_eq!(saved, None, "nothing to save into, so no profile name back");
}

/// The guard must not be over-broad: a name that IS a configured profile still
/// reaches the loader/saver. The test scaffolding's loader always errors, so
/// reaching it at all is the signal — an `Ok(None)` here would mean the guard
/// had swallowed a real profile and silently stopped persisting model choices.
#[test]
fn a_configured_profile_still_reaches_the_persist_path() {
    let mut app = test_app("xai", "grok-4.3");
    app.profiles = vec![crate::ProfileInfo {
        name: "work".into(),
        provider: "xai".into(),
        model: Some("grok-4.3".into()),
        base_url: None,
        managed_local_repo: None,
        managed_local_path: None,
    }];
    app.active_profile = Some("work".to_string());

    assert!(
        app.persist_active_profile_model("grok-4.5").is_err(),
        "a configured profile must go through the loader, not be skipped"
    );
}

#[test]
fn density_cycles_and_compact_folds_tool_bodies() {
    assert_eq!(Density::Comfortable.next(), Density::Verbose);
    assert_eq!(Density::Verbose.next(), Density::Compact);
    assert_eq!(Density::parse("verbose"), Some(Density::Verbose));

    let body: Vec<Line<'static>> = (0..20).map(|i| Line::raw(format!("line {i}"))).collect();
    let entry = TranscriptEntry::ToolOutput {
        body,
        expanded: false,
    };
    let compact = entry.flatten(false, false, Density::Compact);
    assert_eq!(compact.len(), 1, "compact collapses to one fold line");
    assert!(
        crate::render::line_text(&compact[0]).contains("folded"),
        "compact fold labels itself"
    );
    let verbose = entry.flatten(false, false, Density::Verbose);
    assert!(verbose.len() >= 20, "verbose expands the body");
}

#[test]
fn queue_select_reorder_and_remove() {
    let mut app = test_app("openai", "gpt-4o");
    app.queue.push_back("one".into());
    app.queue.push_back("two".into());
    app.queue.push_back("three".into());

    app.queue_select_next();
    assert_eq!(app.queue_selected, Some(0));
    app.queue_select_next();
    assert_eq!(app.queue_selected, Some(1));
    app.queue_move_selected(1);
    assert_eq!(
        app.queue.iter().cloned().collect::<Vec<_>>(),
        vec!["one", "three", "two"]
    );
    assert_eq!(app.queue_selected, Some(2));
    let removed = app.queue_remove_selected();
    assert_eq!(removed.as_deref(), Some("two"));
    assert_eq!(app.queue.len(), 2);
}

#[test]
fn prompt_queue_rejects_past_cap() {
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..crate::MAX_PROMPT_QUEUE {
        assert!(
            app.try_enqueue_prompt(format!("p{i}")),
            "enqueue {i} should fit under the cap"
        );
    }
    assert_eq!(app.queue.len(), crate::MAX_PROMPT_QUEUE);
    assert!(
        !app.try_enqueue_prompt("overflow"),
        "cap must reject further enqueues"
    );
    assert_eq!(app.queue.len(), crate::MAX_PROMPT_QUEUE);
    assert_eq!(app.queue.front().map(String::as_str), Some("p0"));
    assert!(!app.enqueue_prompt("also-overflow"));
    assert!(
        app.transcript_text().contains("prompt queue full"),
        "interactive enqueue should surface a warning"
    );
}

#[test]
fn enqueue_prompt_front_evicts_newest_when_full() {
    let mut app = test_app("openai", "gpt-4o");
    for i in 0..crate::MAX_PROMPT_QUEUE {
        assert!(app.try_enqueue_prompt(format!("p{i}")));
    }
    assert!(app.enqueue_prompt_front("priority"));
    assert_eq!(app.queue.len(), crate::MAX_PROMPT_QUEUE);
    assert_eq!(app.queue.front().map(String::as_str), Some("priority"));
    // Newest tail dropped to make room; oldest non-priority remains next.
    assert_eq!(app.queue.get(1).map(String::as_str), Some("p0"));
    assert!(
        !app.queue
            .iter()
            .any(|p| p == &format!("p{}", crate::MAX_PROMPT_QUEUE - 1)),
        "newest should be evicted"
    );
}

#[test]
fn path_scoped_auto_approve_matches_prefix_only() {
    let mut app = test_app("openai", "gpt-4o");
    app.add_auto_approve_path("src/lib.rs");
    assert!(app.path_auto_approved("src/lib.rs"));
    assert!(app.path_auto_approved("src/main.rs"));
    assert!(!app.path_auto_approved("crates/other/src/lib.rs"));

    let edit = hi_agent::ConfirmationRequest::FileEdit {
        path: "src/foo.rs".into(),
        diff: "+x".into(),
    };
    assert!(app.should_auto_approve(&edit));
    let shell = hi_agent::ConfirmationRequest::ShellMutation {
        command: "rm x".into(),
        cwd: "/tmp".into(),
    };
    assert!(!app.should_auto_approve(&shell));
    app.auto_approve_session = true;
    let ask = hi_agent::ConfirmationRequest::AskUser {
        question: "which API?".into(),
        options: vec!["REST".into(), "gRPC".into()],
    };
    assert!(
        !app.should_auto_approve(&ask),
        "ask_user must never auto-approve"
    );
    assert!(
        !app.should_auto_approve(&shell),
        "session file auto-approve must not cover shell"
    );
    let browser = hi_agent::ConfirmationRequest::External {
        tool: "browser_exec".into(),
        operation_arguments: serde_json::json!({"script": "goto https://example.com"}),
        summary: "goto https://example.com".into(),
        target: String::new(),
        mcp_grant: None,
    };
    assert!(!app.should_auto_approve(&browser));
    let mcp = hi_agent::ConfirmationRequest::External {
        tool: "use_tool".into(),
        operation_arguments: serde_json::json!({
            "server": "github",
            "tool": "create_issue",
            "arguments": {}
        }),
        summary: "{}".into(),
        target: "github.create_issue".into(),
        mcp_grant: Some(("github".into(), "create_issue".into())),
    };
    assert!(!app.should_auto_approve(&mcp));
    app.add_auto_approve_mcp("github".into(), "create_issue".into());
    assert!(app.should_auto_approve(&mcp));
}

#[test]
fn view_cache_skips_rebuild_on_spinner_only_tick() {
    let mut app = test_app("openai", "gpt-4o");
    app.push(Line::raw("hello"));
    app.ensure_view_cache(80, None);
    let generation = app.view_cache.generation;
    let lines = app.view_cache.lines.len();
    // Spinner tick: no transcript change.
    app.spinner = app.spinner.wrapping_add(1);
    app.ensure_view_cache(80, None);
    assert_eq!(app.view_cache.generation, generation);
    assert_eq!(app.view_cache.lines.len(), lines);
    // Structural change busts the cache.
    app.push(Line::raw("world"));
    app.ensure_view_cache(80, None);
    assert_ne!(app.view_cache.generation, generation);
    assert!(app.view_cache.lines.len() > lines);
}

#[test]
fn view_cache_rebuilds_in_place_progress_updates() {
    let mut app = test_app("openai", "gpt-4o");
    let mut slot = None;
    app.push_or_replace_progress(&mut slot, "⟳", Line::raw("⟳ first progress"));
    app.ensure_view_cache(80, None);

    app.push_or_replace_progress(&mut slot, "⟳", Line::raw("⟳ replacement progress"));
    app.ensure_view_cache(80, None);
    assert!(
        app.view_cache
            .lines
            .iter()
            .any(|line| crate::render::line_text(line).contains("replacement progress"))
    );
}

#[test]
fn non_markdown_stream_lines_invalidate_the_view_cache() {
    let mut app = test_app("openai", "gpt-4o");
    app.ensure_view_cache(80, None);
    app.stream(Style::default(), false, "btw result\n");
    app.ensure_view_cache(80, None);
    assert!(
        app.view_cache
            .lines
            .iter()
            .any(|line| crate::render::line_text(line).contains("btw result"))
    );
}

#[test]
fn view_cache_refreshes_the_compacted_line_count() {
    let mut app = test_app("openai", "gpt-4o");
    app.transcript = (0..MAX_TRANSCRIPT_LINES)
        .map(|_| TranscriptEntry::Line(Line::raw("line")))
        .collect();
    app.bump_transcript();
    app.ensure_view_cache(80, None);

    app.push(Line::raw("one more"));
    app.ensure_view_cache(80, None);
    assert!(
        app.view_cache
            .lines
            .first()
            .is_some_and(|line| crate::render::line_text(line).contains("↑ 1 lines"))
    );

    app.push(Line::raw("two more"));
    app.ensure_view_cache(80, None);
    assert!(
        app.view_cache
            .lines
            .first()
            .is_some_and(|line| crate::render::line_text(line).contains("↑ 2 lines"))
    );
}

#[test]
fn uimode_is_exclusive_review_clears_normal() {
    let mut app = test_app("openai", "gpt-4o");
    app.mode = crate::mode::UiMode::Normal { search: None };
    app.open_review(None);
    assert!(app.mode.is_review());
    assert!(!app.mode.is_normal());
}

#[test]
fn action_dispatch_toggles_diff_and_help() {
    use crate::action::{Action, KeySurface, resolve_key};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    let toggle_diff = resolve_key(
        KeySurface::Insert,
        &KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL),
    );
    assert_eq!(toggle_diff, Action::ToggleDiff);

    let mut app = test_app("openai", "gpt-4o");
    app.apply_action(Action::ToggleHelp);
    assert!(app.show_help);
    app.apply_action(Action::ToggleHelp);
    assert!(!app.show_help);
}

#[test]
fn command_palette_filters_and_accepts() {
    use crate::palette::{CommandPalette, PaletteOutcome};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    let mut p = CommandPalette::open();
    assert!(p.items.len() > 3);
    for c in "help".chars() {
        p.insert(c);
    }
    assert!(
        p.current().is_some_and(|i| i.label.contains("help")),
        "expected /help match"
    );
    let out = p.handle_key(&KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert!(matches!(out, PaletteOutcome::Accept(s) if s.contains("help")));
}

#[test]
fn tutorial_overlay_renders_centered_content() {
    let mut app = test_app("openai", "gpt-4o");
    app.tutorial = Some(crate::tutorial::TutorialOverlay::fresh());
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("hi tutorial"),
        "missing modal title: {screen}"
    );
    assert!(
        screen.contains("Lesson 1 of 8"),
        "missing progress: {screen}"
    );
    assert!(
        screen.contains("Ask for outcomes"),
        "missing lesson: {screen}"
    );
    assert!(
        screen.contains("Enter next"),
        "missing navigation: {screen}"
    );
}

#[test]
fn header_shows_ctx_not_settings() {
    let mut app = test_app("openai", "gpt-4o");
    app.density = Density::Compact;
    app.context_used = 38_000;
    app.context_window = Some(1_000_000);
    app.last_turn_latency = Some(Duration::from_secs(65));
    let mut term = Terminal::new(TestBackend::new(80, 20)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("38k / 1.0M"),
        "context chip missing: {screen}"
    );
    assert!(
        !screen.contains("compact")
            && !screen.contains("out:fold")
            && !screen.contains("reasoning"),
        "settings chips should stay off the header: {screen}"
    );
}

#[test]
fn jump_prompt_moves_scroll() {
    let mut app = test_app("openai", "gpt-4o");
    app.push_user_prompt(Line::raw("❯ first"));
    for i in 0..30 {
        app.push(Line::raw(format!("body {i}")));
    }
    app.push_user_prompt(Line::raw("❯ second"));
    for i in 0..30 {
        app.push(Line::raw(format!("more {i}")));
    }
    app.ensure_view_cache(80, None);
    assert!(
        app.view_cache.prompt_line_starts.len() >= 2,
        "expected two user prompts in the cache"
    );
    // Sit past the second prompt, then jump backward.
    let second_row = app.view_cache.prefix[app.view_cache.prompt_line_starts[1]] as u16;
    app.scroll_to(second_row.saturating_add(5));
    let before = app.scroll;
    app.jump_transcript_marker(crate::dispatch::TranscriptMarker::UserPrompt, -1);
    assert!(
        app.scroll <= before,
        "prev-prompt jump should not move downward ({before} -> {})",
        app.scroll
    );
    // And the landing row should be one of the prompt rows.
    let prompt_rows: Vec<u16> = app
        .view_cache
        .prompt_line_starts
        .iter()
        .filter_map(|&i| app.view_cache.prefix.get(i).map(|r| *r as u16))
        .collect();
    assert!(
        prompt_rows.contains(&app.scroll) || app.scroll < before,
        "scroll {} should land on a prompt row {prompt_rows:?}",
        app.scroll
    );
}

#[test]
fn session_chrome_matches_grok_build_stack() {
    // Grok-build's session face: flat status, unboxed scrollback, rounded
    // prompt, shortcuts row. The transcript must not wear a titled rounded box.
    let mut app = test_app("openai", "gpt-4o");
    app.push_user_prompt(Line::raw("❯ hello"));
    app.apply(UiEvent::Text {
        text: "working on it\n".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    app.input.set("next");

    // Auto-compact (≤20 rows) drops the outer vertical pad so chrome sits on
    // the first and last rows.
    let mut compact = Terminal::new(TestBackend::new(80, 20)).unwrap();
    compact.draw(|f| app.render(f)).unwrap();
    let compact_screen = dump(&compact);
    let compact_rows: Vec<&str> = compact_screen.lines().collect();
    assert!(
        compact_rows[0].contains("/workspace") || compact_rows[0].contains("workspace"),
        "compact status bar shows cwd on row 0: {compact_screen}"
    );
    assert!(
        !compact_rows[0].trim_start().starts_with('╭'),
        "status bar is not a titled box: {}",
        compact_rows[0]
    );
    assert!(
        compact_rows.iter().any(|l| l.trim_start().starts_with('╭')),
        "prompt still has a rounded top: {compact_screen}"
    );
    assert!(
        compact_rows
            .iter()
            .any(|l| l.contains("Shift+Tab") && l.contains("mode")),
        "idle session still shows the grok-build mode hint: {compact_screen}"
    );
    assert!(
        compact_screen.contains("❯ next") || compact_screen.contains("next"),
        "composer shows the draft: {compact_screen}"
    );
    let prompt_border = compact_rows
        .iter()
        .rev()
        .find(|l| l.contains('╰') && l.contains("gpt-4o"))
        .expect("prompt bottom border with model");
    let after_corner = prompt_border.trim_start().trim_start_matches('╰');
    let model_at = after_corner.find("gpt-4o").expect("model on bottom border");
    let rule_at = after_corner.find('─').unwrap_or(0);
    assert!(
        model_at > rule_at,
        "model sits on the right of the bottom divider, grok-build style: {prompt_border}"
    );

    // Tall terminals keep a blank canvas row above the status bar and below
    // the shortcuts, matching grok-build's outer_vpad.
    let mut tall = Terminal::new(TestBackend::new(80, 24)).unwrap();
    tall.draw(|f| app.render(f)).unwrap();
    let tall_screen = dump(&tall);
    let tall_rows: Vec<&str> = tall_screen.lines().collect();
    assert!(
        tall_rows[0].trim().is_empty(),
        "tall terminal keeps a blank top pad: {tall_screen}"
    );
    assert!(
        tall_rows[1].contains("/workspace") || tall_rows[1].contains("workspace"),
        "status bar sits under the top pad: {tall_screen}"
    );
    assert!(
        tall_rows.last().is_some_and(|l| l.trim().is_empty()),
        "tall terminal keeps a blank bottom pad: {tall_screen}"
    );
    assert!(
        tall_screen.contains("Shift+Tab:mode"),
        "idle transcript keeps Shift+Tab:mode on tall terminals: {tall_screen}"
    );

    let mut fresh = test_app("openai", "gpt-4o");
    let mut welcome = Terminal::new(TestBackend::new(80, 20)).unwrap();
    welcome.draw(|f| fresh.render(f)).unwrap();
    let welcome_screen = dump(&welcome);
    assert!(
        welcome_screen.contains("Shift+Tab") && welcome_screen.contains("mode"),
        "empty session shows mode cycle hint: {welcome_screen}"
    );
}

#[test]
fn theme_roles_not_raw_ansi_in_session_render() {
    // Guard: session render path should not reintroduce Color::Yellow etc.
    // (dashboard/watch still may during migration — this checks the string
    // source of app/render.rs at unit-test time via include_str).
    let src = include_str!("app/render.rs");
    for bad in [
        "Color::Yellow",
        "Color::Cyan",
        "Color::Green",
        "Color::Red",
        "Color::Magenta",
    ] {
        assert!(
            !src.contains(bad),
            "app/render.rs still contains {bad} — use theme roles"
        );
    }
}

#[test]
fn subagent_spawn_updates_one_live_row_then_finishes() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::SubagentSpawned {
        id: "explore-1".into(),
        subagent_kind: "explore".into(),
        description: "crate boundaries".into(),
        background: false,
    });
    let text = app.transcript_text();
    assert!(
        text.contains("Explore") && text.contains("crate boundaries"),
        "spawned row missing: {text}"
    );
    assert!(!text.contains("explore:read"));
    app.apply(UiEvent::SubagentProgress {
        id: "explore-1".into(),
        activity: "Reading lib.rs".into(),
        line: Some("read lib.rs".into()),
    });
    let text = app.transcript_text();
    assert!(
        text.contains("Reading lib.rs"),
        "live suffix missing: {text}"
    );
    assert_eq!(
        app.transcript
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Activity(_)))
            .count(),
        1,
        "foreground explore must stay one row"
    );
    app.apply(UiEvent::SubagentFinished {
        id: "explore-1".into(),
        status: "completed".into(),
        elapsed_ms: 12_000,
        summary: "done".into(),
    });
    let text = app.transcript_text();
    assert!(
        text.contains("completed") && text.contains("crate boundaries"),
        "finish should update the same row: {text}"
    );
    assert_eq!(
        app.transcript
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Activity(_)))
            .count(),
        1
    );
}

#[test]
fn subagent_inspect_line_keeps_live_suffix() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::SubagentSpawned {
        id: "delegate-1".into(),
        subagent_kind: "delegate".into(),
        description: "fix the bug".into(),
        background: false,
    });
    app.apply(UiEvent::SubagentProgress {
        id: "delegate-1".into(),
        activity: "Reading lib.rs".into(),
        line: Some("Reading lib.rs".into()),
    });
    app.apply(UiEvent::SubagentProgress {
        id: "delegate-1".into(),
        activity: String::new(),
        line: Some("fn foo() {}".into()),
    });
    let text = app.transcript_text();
    assert!(
        text.contains("Reading lib.rs"),
        "empty-activity inspect lines must keep the live suffix: {text}"
    );
    assert!(
        !text.contains("explore:read") && !text.contains("fn foo()"),
        "inspect body must not dump into the parent feed: {text}"
    );
    assert_eq!(
        app.subagents.get("delegate-1").map(|info| info.lines.len()),
        Some(2)
    );
}

#[test]
fn subagent_background_task_is_two_rows() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::SubagentSpawned {
        id: "task_1".into(),
        subagent_kind: "explore".into(),
        description: "scan deps".into(),
        background: true,
    });
    app.apply(UiEvent::SubagentProgress {
        id: "task_1".into(),
        activity: "Reading Cargo.toml".into(),
        line: None,
    });
    let start = app.transcript_text();
    assert!(start.contains("started"), "background start row: {start}");
    assert!(
        !start.contains("Reading Cargo.toml"),
        "bg start row stays started: {start}"
    );
    app.apply(UiEvent::SubagentFinished {
        id: "task_1".into(),
        status: "completed".into(),
        elapsed_ms: 4000,
        summary: "ok".into(),
    });
    let text = app.transcript_text();
    assert!(text.contains("started") && text.contains("completed"));
    assert_eq!(
        app.transcript
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Activity(_)))
            .count(),
        2,
        "background finish appends a second row: {text}"
    );
}

#[test]
fn subagent_tool_calls_do_not_dump_explore_reads() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::SubagentSpawned {
        id: "explore-1".into(),
        subagent_kind: "explore".into(),
        description: "find X".into(),
        background: false,
    });
    app.apply(UiEvent::ToolCall {
        name: "explore".into(),
        arguments: r#"{"task":"find X"}"#.into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "explore:read".into(),
        arguments: r#"{"path":"lib.rs"}"#.into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "explore:read".into(),
        result: "fn main() {}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "explore".into(),
        result: "found it".into(),
    });
    let text = app.transcript_text();
    assert!(
        !text.contains("explore:read") && !text.contains("explore:grep"),
        "parent feed dumped child tools: {text}"
    );
    assert!(text.contains("Explore") && text.contains("find X"));
}

#[test]
fn waiting_on_subagent_chrome() {
    let mut app = test_app("openai", "gpt-4o");
    app.working = true;
    app.apply(UiEvent::SubagentSpawned {
        id: "explore-1".into(),
        subagent_kind: "explore".into(),
        description: "map crate".into(),
        background: false,
    });
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Waiting on subagent"),
        "blocking chrome missing: {screen}"
    );
}

#[test]
fn background_subagents_still_running_chrome() {
    let mut app = test_app("openai", "gpt-4o");
    app.working = false;
    app.apply(UiEvent::SubagentSpawned {
        id: "task_1".into(),
        subagent_kind: "explore".into(),
        description: "scan".into(),
        background: true,
    });
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("subagent still running"),
        "idle bg chrome missing: {screen}"
    );
}

#[test]
fn inspect_overlay_opens_from_feed_row_and_closes() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::SubagentSpawned {
        id: "explore-1".into(),
        subagent_kind: "explore".into(),
        description: "crate boundaries".into(),
        background: false,
    });
    app.apply(UiEvent::SubagentProgress {
        id: "explore-1".into(),
        activity: "Reading lib.rs".into(),
        line: Some("read lib.rs".into()),
    });
    crate::subagent_overlay::open_inspect(&mut app, "explore-1");
    assert!(app.inspect_subagent.is_some());
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("crate boundaries") && screen.contains("read lib.rs"),
        "inspect overlay missing child lines: {screen}"
    );
    let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
    match crate::subagent_overlay::handle_inspect_key(&mut app, &esc) {
        crate::subagent_overlay::OverlayOutcome::Close => app.inspect_subagent = None,
        _ => panic!("esc should close inspect"),
    }
    assert!(app.inspect_subagent.is_none());
}

#[test]
fn tasks_overlay_lists_running_id() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::SubagentSpawned {
        id: "task_1".into(),
        subagent_kind: "explore".into(),
        description: "scan deps".into(),
        background: true,
    });
    crate::subagent_overlay::open_tasks(&mut app, &[], &["task_1".into()]);
    let overlay = app.tasks_overlay.as_ref().expect("tasks overlay");
    assert!(
        overlay.rows.iter().any(|row| row.id == "task_1"),
        "running id missing: {:?}",
        overlay.rows.iter().map(|r| &r.id).collect::<Vec<_>>()
    );
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("TASKS") && screen.contains("scan deps"),
        "tasks overlay: {screen}"
    );
}

#[test]
fn turn_status_working_shows_stop_not_ctrl_c() {
    let mut app = test_app("openai", "gpt-4o");
    app.working = true;
    let line = crate::turn_status::build(&app, 80).expect("working strip");
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    assert!(text.contains("[stop]"), "{text}");
    assert!(!text.contains("Ctrl-C"), "{text}");
}

#[test]
fn turn_status_confirmation_waits_on_you() {
    let mut app = test_app("openai", "gpt-4o");
    app.confirmation = Some(hi_agent::ConfirmationRequest::ShellMutation {
        command: "rm x".into(),
        cwd: "/tmp".into(),
    });
    let line = crate::turn_status::build(&app, 80).expect("waiting strip");
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    assert!(text.contains("Waiting on you"), "{text}");
}

#[test]
fn ctrl_f_opens_block_viewer_with_expanded_hunks() {
    use crate::action::Action;
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"src/cli.rs\",\"old_string\":\"a\",\"new_string\":\"b\"}".into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "--- a/src/cli.rs\n+++ b/src/cli.rs\n@@ -1,1 +1,1 @@\n-old\n+new\n".into(),
    });
    app.apply_action(Action::OpenBlockViewer);
    let viewer = app.block_viewer.as_ref().expect("block viewer");
    let text = viewer.texts.join("\n");
    assert!(
        text.contains("@@") && (text.contains("+new") || text.contains("new")),
        "expanded hunk missing: {text}"
    );
    let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
    assert!(matches!(
        crate::block_viewer::handle_key(&mut app, &esc),
        crate::block_viewer::ViewerOutcome::Close
    ));
}

#[test]
fn jump_picker_scrolls_and_esc_restores() {
    let mut app = test_app("openai", "gpt-4o");
    app.push_user_prompt(Line::raw("first prompt"));
    for i in 0..40 {
        app.transcript
            .push(TranscriptEntry::Assistant(Line::raw(format!("pad {i}"))));
    }
    app.push_user_prompt(Line::raw("second prompt"));
    app.view_max_scroll = 200;
    app.scroll = 12;
    app.following = false;
    let restore = app.scroll;
    app.open_jump_picker();
    assert!(app.jump_picker.is_some());
    let key = KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE);
    crate::session_pickers::handle_jump_key(&mut app, &key);
    assert_ne!(app.scroll, restore, "j/k should live-scroll");
    let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
    crate::session_pickers::handle_jump_key(&mut app, &esc);
    assert!(app.jump_picker.is_none());
    assert_eq!(app.scroll, restore);
}

#[test]
fn rewind_transcript_drops_chosen_prompt_and_later_rows() {
    let mut app = test_app("openai", "gpt-4o");
    app.push_user_prompt(Line::raw("keep me"));
    app.transcript
        .push(TranscriptEntry::Assistant(Line::raw("kept answer")));
    app.push_user_prompt(Line::raw("drop me"));
    app.transcript
        .push(TranscriptEntry::Assistant(Line::raw("later answer")));
    app.rewind_transcript_to_user_turn(2);
    let text: String = app.transcript.iter().map(TranscriptEntry::text).collect();
    assert!(text.contains("keep me"), "{text}");
    assert!(text.contains("kept answer"), "{text}");
    assert!(!text.contains("drop me"), "{text}");
    assert!(!text.contains("later answer"), "{text}");
}

#[test]
fn rewind_picker_confirm_emits_turn_number() {
    let mut app = test_app("openai", "gpt-4o");
    app.rewind_picker = crate::session_pickers::RewindPicker::new(vec![
        hi_agent::UserTurn {
            n: 1,
            message_index: 1,
            preview: "one".into(),
        },
        hi_agent::UserTurn {
            n: 2,
            message_index: 3,
            preview: "two".into(),
        },
    ]);
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(
        crate::session_pickers::handle_rewind_key(&mut app, &enter),
        crate::session_pickers::PickerOutcome::Continue
    ));
    assert!(matches!(
        crate::session_pickers::handle_rewind_key(&mut app, &enter),
        crate::session_pickers::PickerOutcome::Rewind(2)
    ));
}

#[test]
fn timeline_rail_ticks_with_two_prompts() {
    let mut app = test_app("openai", "gpt-4o");
    app.timeline_enabled = true;
    app.push_user_prompt(Line::raw("turn one"));
    app.transcript
        .push(TranscriptEntry::Assistant(Line::raw("answer one")));
    app.push_user_prompt(Line::raw("turn two"));
    app.transcript
        .push(TranscriptEntry::Assistant(Line::raw("answer two")));
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert_eq!(app.timeline_rect.width, 2);
    assert!(
        !app.timeline_hits.is_empty(),
        "expected rail hits: {screen}"
    );
    assert!(
        screen.contains('·') || screen.contains('●'),
        "rail ticks missing: {screen}"
    );
}

#[tokio::test]
async fn rewind_without_arg_opens_picker() {
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
    agent
        .apply_loaded_session(
            vec![
                hi_ai::Message::system("sys"),
                hi_ai::Message::user("first"),
                hi_ai::Message::assistant(vec![hi_ai::Content::Text("a1".into())]),
                hi_ai::Message::user("second"),
                hi_ai::Message::assistant(vec![hi_ai::Content::Text("a2".into())]),
            ],
            hi_ai::Usage::default(),
            Vec::new(),
            None,
            hi_agent::DecisionLog::default(),
            Vec::new(),
        )
        .unwrap();
    let mut app = test_app("openai", "gpt-4o");
    app.handle_command(&mut agent, hi_agent::Command::Rewind(String::new()))
        .await;
    let picker = app.rewind_picker.as_ref().expect("rewind picker");
    assert_eq!(picker.turns.len(), 2);
}

#[test]
fn composer_shows_always_approve_flag_by_default() {
    let mut app = test_app("openai", "gpt-4o");
    let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("always-approve"),
        "default Always face is flagged on the prompt:\n{screen}"
    );
}

#[test]
fn plan_mode_paints_plan_flag_on_the_composer() {
    let mut app = test_app("openai", "gpt-4o");
    app.plan_mode = true;
    app.permission_mode = hi_agent::PermissionMode::Ask;
    let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("plan"), "plan flag shown:\n{screen}");
    assert!(
        !screen.contains("always-approve"),
        "always-approve hidden while planning:\n{screen}"
    );
}

#[test]
fn leaving_plan_with_leftover_opens_approval_card() {
    let mut app = test_app("openai", "gpt-4o");
    app.plan_mode = true;
    app.permission_mode = hi_agent::PermissionMode::Ask;
    app.plan = vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.cycle_session_face();
    assert_eq!(app.session_face(), crate::session_face::SessionFace::Always);
    assert!(app.plan_approval.is_some());
}

#[test]
fn plan_approval_card_renders_choices() {
    let mut app = test_app("openai", "gpt-4o");
    app.plan = vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.open_plan_approval();
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("Plan approval"), "{screen}");
    assert!(screen.contains("Approve"), "{screen}");
    assert!(screen.contains("Request changes"), "{screen}");
    assert!(screen.contains("wire the scheduler"), "{screen}");
}

#[test]
fn queue_pane_shows_numbered_rows_above_the_prompt() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.queue.push_back("run the tests".into());
    app.queue.push_back("then commit".into());
    let mut term = Terminal::new(TestBackend::new(60, 16)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("#1 run the tests"), "{screen}");
    assert!(screen.contains("#2 then commit"), "{screen}");
}

#[test]
fn confirm_overlay_highlights_the_first_option() {
    let mut app = test_app("openai", "gpt-4o");
    app.confirmation = Some(hi_agent::ConfirmationRequest::ShellMutation {
        command: "rm generated.txt".into(),
        cwd: "/workspace".into(),
    });
    let mut term = Terminal::new(TestBackend::new(120, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("Approve once"), "{screen}");
    assert!(
        !screen.contains("Always allow this session"),
        "shell must not offer standing allow: {screen}"
    );
    assert!(screen.contains("Reject and follow up"), "{screen}");
}

#[test]
fn plan_approval_esc_parks_and_turn_status_unparks() {
    use crossterm::event::{
        KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    };
    let mut app = test_app("openai", "gpt-4o");
    app.plan = vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.open_plan_approval();
    assert!(app.plan_approval_capturing());
    let outcome = crate::plan_approval::handle_key(
        &mut app,
        &KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
    );
    assert_eq!(outcome, crate::plan_approval::PlanApprovalOutcome::Continue);
    assert!(!app.plan_approval_capturing());
    assert!(app.plan_approval.as_ref().is_some_and(|c| c.parked));

    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("/view-plan"), "{screen}");
    assert!(
        !screen.contains("Approve — leave"),
        "parked card leaves the composer: {screen}"
    );

    let rect = app.turn_status_rect;
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: rect.x,
        row: rect.y,
        modifiers: KeyModifiers::NONE,
    });
    assert!(
        app.plan_approval_capturing(),
        "clicking the turn-status row reopens the card"
    );
}

#[test]
fn plan_comments_seed_request_changes() {
    let mut app = test_app("openai", "gpt-4o");
    app.plan = vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.open_plan_approval();
    app.plan_approval
        .as_mut()
        .unwrap()
        .comments
        .push(crate::plan_approval::PlanComment {
            step: 0,
            text: "too vague".into(),
        });
    app.apply_plan_request_changes_local();
    assert!(app.plan_mode);
    assert!(
        app.input.text().contains("too vague") && app.input.text().contains("wire the scheduler"),
        "comments seed the composer: {}",
        app.input.text()
    );
}

#[test]
fn welcome_home_paints_repo_branch_and_sessions() {
    let mut app = test_app("openai", "gpt-4o");
    app.git_branch = Some("main".into());
    app.session_completion_cache = vec![crate::LocalSessionInfo {
        id: "s1".into(),
        title: "review layout".into(),
        age: "2h".into(),
        lines: 4,
    }];
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("workspace:main"), "{screen}");
    assert!(screen.contains("review layout"), "{screen}");
    assert!(screen.contains("Shift-Tab plan"), "{screen}");
}

#[test]
fn live_task_strip_lists_running_subagents() {
    let mut app = test_app("openai", "gpt-4o");
    app.set_working(true);
    app.apply(UiEvent::SubagentSpawned {
        id: "explore-1".into(),
        subagent_kind: "explore".into(),
        description: "crate boundaries".into(),
        background: false,
    });
    app.apply(UiEvent::SubagentSpawned {
        id: "task-1".into(),
        subagent_kind: "task".into(),
        description: "run tests".into(),
        background: true,
    });
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("Subagents"), "{screen}");
    assert!(screen.contains("explore crate boundaries"), "{screen}");
    assert!(screen.contains("task run tests"), "{screen}");
}

#[test]
fn memory_browser_lists_project_and_global() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let mut app = test_app("openai", "gpt-4o");
    app.memory_browser = Some(crate::memory_browser::MemoryBrowser::open(
        &app.workspace_root,
    ));
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("project"), "{screen}");
    assert!(screen.contains("global"), "{screen}");
    crate::memory_browser::handle_key(&mut app, &KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(app.memory_browser.is_none());
}

#[test]
fn context_chip_hover_swaps_in_place() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Usage {
        prompt: 12,
        generated: 340,
        ctx_used: 64_000,
        ctx_window: Some(128_000),
        estimated: false,
    });
    let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let idle = dump(&term);
    assert!(idle.contains("64k / 128k"), "{idle}");
    app.mouse_col = app.ctx_chip_rect.x;
    app.mouse_row = app.ctx_chip_rect.y;
    term.draw(|f| app.render(f)).unwrap();
    let hover = dump(&term);
    assert!(
        hover.contains("#####------") || hover.contains('#'),
        "{hover}"
    );
    assert!(!hover.contains("64k / 128k"), "{hover}");
}

#[test]
fn permission_auto_approves_only_safe_edits() {
    let mut app = test_app("openai", "gpt-4o");
    app.permission_mode = hi_agent::PermissionMode::Auto;
    let safe = hi_agent::ConfirmationRequest::FileEdit {
        path: "src/lib.rs".into(),
        diff: "+fn ok() {}\n".into(),
    };
    let secret = hi_agent::ConfirmationRequest::FileEdit {
        path: ".env".into(),
        diff: "+TOKEN=x\n".into(),
    };
    let shell = hi_agent::ConfirmationRequest::ShellMutation {
        command: "rm x".into(),
        cwd: "/tmp".into(),
    };
    let ask = hi_agent::ConfirmationRequest::AskUser {
        question: "which API?".into(),
        options: vec!["REST".into()],
    };
    assert!(app.should_auto_approve(&safe));
    assert!(!app.should_auto_approve(&secret));
    assert!(!app.should_auto_approve(&shell));
    assert!(!app.should_auto_approve(&ask));
    let browser = hi_agent::ConfirmationRequest::External {
        tool: "browser_exec".into(),
        operation_arguments: serde_json::json!({"script": "goto https://example.com"}),
        summary: "goto https://example.com".into(),
        target: String::new(),
        mcp_grant: None,
    };
    assert!(!app.should_auto_approve(&browser));
}

#[test]
fn confirm_hint_mentions_queued_permissions() {
    let request = hi_agent::ConfirmationRequest::ShellMutation {
        command: "rm x".into(),
        cwd: "/tmp".into(),
    };
    let hint =
        crate::confirm_overlay::hint(&request, crate::confirm_overlay::ConfirmFocus::Options, 2);
    assert!(hint.contains("2 waiting"), "{hint}");
}

#[test]
fn path_completion_inserts_without_trailing_space_so_ranges_work() {
    let mut app = test_app("openai", "gpt-4o");
    app.path_completion_cache = vec!["src/lib.rs".into()];
    app.completion = Some(crate::completion::CompletionState {
        ctx: crate::completion::CompletionContext::Path {
            prefix: String::new(),
        },
        selected: 0,
    });
    app.input.set("@lib");
    app.sync_completion();
    let labels: Vec<String> = app
        .completion_items()
        .iter()
        .map(|i| i.label.clone())
        .collect();
    assert!(
        labels.iter().any(|l| l == "src/lib.rs"),
        "fuzzy @ menu: {labels:?}"
    );
    assert_eq!(app.accept_completion(false), None);
    assert_eq!(app.input.text(), "@src/lib.rs");
    app.input.set("@src/lib.rs:40");
    app.sync_completion();
    assert!(
        app.completion.is_none(),
        "typing :range closes the path menu"
    );
}
