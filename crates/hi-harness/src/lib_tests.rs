use super::*;
use crate::pipe::test_support::{MockPipe, Scripted, text_chunk, tool_chunk, usage_chunk};
use hi_tools::ProcessRunner;
use hi_tools::sandbox::SandboxPolicy;
use std::fs;

fn test_harness(url: &str, workspace: PathBuf) -> Harness {
    let state = workspace.join(".hi");
    let runner = ProcessRunner::new_with_policy(&workspace, SandboxPolicy::Off).expect("runner");
    let tools = ToolHost::new_with_runner(workspace.clone(), state.clone(), runner).unwrap();
    let mut config = HarnessConfig::pipe(workspace, "pk_test");
    config.base_url = url.to_string();
    config.state_root = state;
    Harness::new_with_tools(config, tools).unwrap()
}

#[tokio::test]
async fn text_only_turn_completes() {
    let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
        text_chunk("all good"),
        usage_chunk(4, 2),
    ])]) else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let mut ui = TestUi {
        confirm: ConfirmationResult::Approved,
        ..TestUi::default()
    };
    let outcome = harness
        .run_turn_cancellable("hello", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(ui.texts.join("").contains("all good"));
    assert_eq!(harness.messages().len(), 2);
    assert_eq!(harness.session_usage().input_tokens, 4);
    assert_eq!(harness.session_usage().output_tokens, 2);
    assert_eq!(harness.usage_snapshot().usage.input_tokens, 4);
}

#[tokio::test]
async fn write_tool_then_final_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("note.txt");
    let args = serde_json::json!({"path": "note.txt", "content": "hi from pipe"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_w", "write", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![text_chunk("wrote it"), usage_chunk(10, 3)]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("write a note", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert_eq!(fs::read_to_string(&path).unwrap(), "hi from pipe");
    assert!(outcome.changed_files.iter().any(|f| f.contains("note.txt")));
}

#[tokio::test]
async fn long_tool_loops_are_not_round_capped() {
    let dir = tempfile::tempdir().unwrap();
    let mut scripts = Vec::new();
    for i in 0..70 {
        // Vary arguments so this is a long loop, not an identical-tool storm.
        let args = serde_json::json!({"path": format!("p{i}")}).to_string();
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_{i}"), "list", &args),
            usage_chunk(1, 1),
        ]));
    }
    scripts.push(Scripted::Sse(vec![
        text_chunk("finished the long job"),
        usage_chunk(2, 2),
    ]));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("keep going", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(ui.texts.join("").contains("finished the long job"));
    assert!(
        ui.tool_calls
            .iter()
            .filter(|(name, _)| name == "list")
            .count()
            >= 70,
        "expected 70 list rounds, got {:?}",
        ui.tool_calls.len()
    );
}

#[tokio::test]
async fn identical_tool_storm_stops_the_turn() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let mut scripts = Vec::new();
    for i in 0..hi_liveness::IDENTICAL_TOOL_CONSECUTIVE {
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_s{i}"), "list", &args),
            usage_chunk(1, 1),
        ]));
    }
    scripts.push(Scripted::Sse(vec![
        text_chunk("should not be requested"),
        usage_chunk(1, 1),
    ]));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("loop list", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.errors.iter().any(|(kind, _)| kind == "tool_storm"),
        "storm must surface a turn error, got {:?}",
        ui.errors
    );
    assert_eq!(
        harness.liveness().snapshot().invariant.map(|inv| inv.code),
        Some(hi_liveness::InvariantCode::IdenticalToolStorm)
    );
}

#[tokio::test]
async fn ask_mode_rejects_write() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "secret.txt", "content": "nope"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_w", "write", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![text_chunk("ok"), usage_chunk(4, 1)]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Ask);
    let mut ui = TestUi {
        confirm: ConfirmationResult::Rejected,
        ..TestUi::default()
    };
    let _ = harness
        .run_turn_cancellable("write", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert!(!dir.path().join("secret.txt").exists());
    assert!(
        ui.tool_results
            .iter()
            .any(|(_, result)| result.contains("rejected"))
    );
}

#[tokio::test]
async fn undo_restores_written_file() {
    let dir = tempfile::tempdir().unwrap();
    hi_test_utils::git::init_git_repo(dir.path());
    fs::write(dir.path().join("keep.txt"), "original").unwrap();
    hi_test_utils::git::git_commit_all(dir.path(), "init");
    let args = serde_json::json!({"path": "keep.txt", "content": "changed"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_w", "write", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![text_chunk("done"), usage_chunk(4, 1)]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    harness
        .run_turn_cancellable("change keep.txt", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(
        fs::read_to_string(dir.path().join("keep.txt")).unwrap(),
        "changed"
    );
    assert!(harness.checkpoint_count() > 0);
    harness.undo().await.unwrap();
    assert_eq!(
        fs::read_to_string(dir.path().join("keep.txt")).unwrap(),
        "original"
    );
}

#[tokio::test]
async fn cancel_during_stream_does_not_hang() {
    let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
        text_chunk("partial"),
        usage_chunk(1, 1),
    ])]) else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let cancel = TurnCancellation::new();
    cancel.cancel();
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("hi", &mut ui, cancel)
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Cancelled);
}

#[test]
fn live_settings_update_while_another_handle_is_held() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = test_harness("http://127.0.0.1:9", dir.path().to_path_buf());
    let live = harness.live();
    live.set_permission_mode(PermissionMode::Always);
    live.apply_effort_arg(EffortArg::Level(ReasoningEffort::High));
    live.set_model("pipe/custom".into());
    assert_eq!(harness.permission_mode(), PermissionMode::Always);
    assert_eq!(harness.reasoning_effort(), Some(ReasoningEffort::High));
    assert_eq!(harness.model(), "pipe/custom");
    harness.set_permission_mode(PermissionMode::Ask);
    assert_eq!(live.permission_mode(), PermissionMode::Ask);
}

#[test]
fn session_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut session = JsonlSession::create(&path).unwrap();
    session.record_messages(&[Message::user("hello")]).unwrap();
    session
        .record_checkpoints(&["sealed:v1:1:ab".into()])
        .unwrap();
    let loaded = JsonlSession::load(&path).unwrap();
    assert_eq!(loaded.messages.len(), 1);
    assert_eq!(loaded.checkpoints.len(), 1);
}

#[test]
fn opening_an_existing_session_does_not_clobber_its_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut session = JsonlSession::create(&path).unwrap();
    session.record_model("pipe/kept").unwrap();
    session.record_messages(&[Message::user("hello")]).unwrap();
    drop(session);

    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.model = "pipe/default".into();
    config.session_path = Some(path.clone());
    config.state_root = dir.path().join(".hi");
    let mut harness = Harness::new(config).unwrap();
    let loaded = JsonlSession::load(&path).unwrap();
    harness.apply_loaded_session(loaded);
    assert_eq!(harness.model(), "pipe/kept");
    assert_eq!(harness.messages().len(), 1);
}

#[test]
fn loaded_pending_turn_does_not_fire_turn_unclosed_on_drop() {
    let dir = tempfile::tempdir().unwrap();
    let publisher = hi_liveness::Publisher::new();
    {
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        config.liveness = Some(publisher.clone());
        let mut harness = Harness::new(config).unwrap();
        harness.apply_loaded_session(LoadedSession {
            messages: vec![Message::user("in flight")],
            pending_turn: Some(crate::PendingTurn {
                turn_index: 3,
                started_unix_ms: 1,
                pre_checkpoint: None,
            }),
            ..LoadedSession::default()
        });
        assert!(harness.pending_turn().is_some());
    }
    assert!(
        publisher.snapshot().invariant.is_none(),
        "loaded PendingTurn must not stamp turn_unclosed"
    );
}

#[test]
fn dropping_an_open_turn_sets_turn_unclosed() {
    let dir = tempfile::tempdir().unwrap();
    let publisher = hi_liveness::Publisher::new();
    {
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        config.liveness = Some(publisher.clone());
        let mut harness = Harness::new(config).unwrap();
        harness.messages.push(Message::user("open"));
        let _ = harness.begin_persisted_turn("open", None);
    }
    assert_eq!(
        publisher.snapshot().invariant.unwrap().code,
        hi_liveness::InvariantCode::TurnUnclosed
    );
}

#[test]
fn clear_history_rewrites_the_session_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.session_path = Some(path.clone());
    config.state_root = dir.path().join(".hi");
    let mut harness = Harness::new(config).unwrap();
    harness.apply_loaded_session(LoadedSession {
        messages: vec![Message::user("old")],
        ..LoadedSession::default()
    });
    harness.clear_history();
    let loaded = JsonlSession::load(&path).unwrap();
    assert!(loaded.messages.is_empty());
}

#[tokio::test]
async fn empty_stream_is_retried_then_finishes() {
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![usage_chunk(4, 0)]),
        Scripted::Sse(vec![text_chunk("reviewed and fixed"), usage_chunk(6, 4)]),
    ]) else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("review and fix", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(ui.texts.join("").contains("reviewed and fixed"));
    assert!(
        harness
            .messages()
            .iter()
            .all(|message| !message.text().is_empty() || message.role != hi_ai::Role::Assistant),
        "empty assistant messages must not be persisted"
    );
}

#[tokio::test]
async fn tool_then_empty_stream_retries_for_final_text() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_l", "list", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![usage_chunk(10, 0)]),
        Scripted::Sse(vec![text_chunk("listing looks fine"), usage_chunk(12, 3)]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("review", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(ui.texts.join("").contains("listing looks fine"));
}

#[tokio::test]
async fn empty_stop_ends_the_turn_without_a_fake_assistant() {
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![usage_chunk(1, 0)]),
        Scripted::Sse(vec![usage_chunk(1, 0)]),
        Scripted::Sse(vec![usage_chunk(1, 0)]),
        Scripted::Sse(vec![usage_chunk(1, 0)]),
    ]) else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("review and fix", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(ui.errors.is_empty());
    assert_eq!(
        harness.messages().len(),
        1,
        "empty stop must not persist an assistant row"
    );
}

fn empty_sse() -> Scripted {
    Scripted::Sse(vec![usage_chunk(10, 0)])
}

/// Pipe retries an empty stream 4 times (attempts 0..=3). One harness
/// continuation is another 4. Two continuations plus the first empty
/// completion is 12 empty scripts after the tool round.
fn empty_stream_retries() -> Vec<Scripted> {
    vec![empty_sse(), empty_sse(), empty_sse(), empty_sse()]
}

#[tokio::test]
async fn empty_stop_after_tools_continues_then_errors_and_reports_invariant() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let mut scripts = vec![Scripted::Sse(vec![
        tool_chunk(0, "call_l", "list", &args),
        usage_chunk(8, 4),
    ])];
    // First empty completion + two harness continuations, each with Pipe's
    // four empty-stream retries.
    for _ in 0..3 {
        scripts.extend(empty_stream_retries());
    }
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("review and fix", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(ui.texts.is_empty());
    assert!(
        ui.errors.iter().any(|(kind, _)| kind == "empty_stop"),
        "empty-after-tools must surface a turn error, got {:?}",
        ui.errors
    );
    assert_eq!(
        harness.liveness().snapshot().invariant.map(|inv| inv.code),
        Some(hi_liveness::InvariantCode::EmptyAssistantAfterTools),
        "Sentinel must see a sticky auto-repair invariant on the live child"
    );
    assert!(
        harness.messages().iter().all(|message| {
            message.role != hi_ai::Role::Assistant || !message.content.is_empty()
        }),
        "empty assistant stop is not persisted"
    );
}

#[tokio::test]
async fn empty_stop_after_tools_recovers_on_continuation() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let mut scripts = vec![Scripted::Sse(vec![
        tool_chunk(0, "call_l", "list", &args),
        usage_chunk(8, 4),
    ])];
    scripts.extend(empty_stream_retries());
    scripts.push(Scripted::Sse(vec![
        text_chunk("listing looks fine"),
        usage_chunk(12, 3),
    ]));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("review and fix", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(ui.texts.join("").contains("listing looks fine"));
    assert!(
        harness.liveness().snapshot().invariant.is_none(),
        "a recovered continuation must not stamp the auto-repair invariant"
    );
}

#[tokio::test]
async fn second_probe_refusal_stops_the_turn_without_a_fifth_pipe_request() {
    let dir = tempfile::tempdir().unwrap();
    let probe = serde_json::json!({
        "command": "./target/debug/chat > /tmp/out.txt &\nsleep 0.05\necho PORT=1"
    })
    .to_string();
    let mut scripts = Vec::new();
    for i in 0..4 {
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_p{i}"), "bash", &probe),
            usage_chunk(8, 4),
        ]));
    }
    // A fifth request would consume this and fail the assertion below.
    scripts.push(Scripted::Sse(vec![
        text_chunk("should not be requested"),
        usage_chunk(1, 1),
    ]));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable(
            "review for any major issues and fix",
            &mut ui,
            TurnCancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.turn_ends
            .iter()
            .any(|end| end.contains("stopped repeating a detached binary probe")),
        "turn_ends={:?}",
        ui.turn_ends
    );
    let requests = server.bodies.lock().unwrap().len();
    assert_eq!(
        requests, 4,
        "four probe rounds then stop; no fifth Pipe request, got {requests}"
    );
    assert!(
        !ui.texts.join("").contains("should not be requested"),
        "fifth script must not run"
    );
}

fn script_four_bash_probes(command: &str) -> Vec<Scripted> {
    let probe = serde_json::json!({ "command": command }).to_string();
    let mut scripts = Vec::new();
    for i in 0..4 {
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_p{i}"), "bash", &probe),
            usage_chunk(8, 4),
        ]));
    }
    scripts.push(Scripted::Sse(vec![
        text_chunk("should not be requested"),
        usage_chunk(1, 1),
    ]));
    scripts
}

async fn assert_probe_turn_stops(command: &str) {
    let dir = tempfile::tempdir().unwrap();
    let Some(server) = MockPipe::new(script_four_bash_probes(command)) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable(
            "review for any major issues and fix",
            &mut ui,
            TurnCancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.turn_ends
            .iter()
            .any(|end| end.contains("stopped repeating a detached binary probe")),
        "turn_ends={:?} command={command}",
        ui.turn_ends
    );
    let requests = server.bodies.lock().unwrap().len();
    assert_eq!(
        requests, 4,
        "four probe rounds then stop; no fifth Pipe request, got {requests} command={command}"
    );
}

#[tokio::test]
async fn cargo_run_sleep_probes_stop_the_turn() {
    assert_probe_turn_stops(
        "CHAT_ADDR=127.0.0.1:0 cargo run >/tmp/srv.log 2>/tmp/srv.err &\nsleep 0.05\ncat /tmp/srv.log",
    )
    .await;
}

#[tokio::test]
async fn python_popen_sleep_probes_stop_the_turn() {
    assert_probe_turn_stops(
        r#"python3 -c 'import subprocess,time; subprocess.Popen(["./target/debug/chat"]); time.sleep(0.05)'"#,
    )
    .await;
}

#[tokio::test]
async fn persist_at_start_does_not_duplicate_user_line_on_finish() {
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("session.jsonl");
    let args = serde_json::json!({"command": "cat session.jsonl"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_c", "bash", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![text_chunk("done"), usage_chunk(10, 3)]),
    ]) else {
        return;
    };
    let state = dir.path().join(".hi");
    let runner = ProcessRunner::new_with_policy(dir.path(), SandboxPolicy::Off).expect("runner");
    let tools = ToolHost::new_with_runner(dir.path().to_path_buf(), state.clone(), runner).unwrap();
    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.base_url = server.url.clone();
    config.state_root = state;
    config.session_path = Some(session_path.clone());
    let mut harness = Harness::new_with_tools(config, tools).unwrap();
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("unique-prompt-xyz", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.tool_results
            .iter()
            .any(|(_, result)| result.contains("unique-prompt-xyz")
                && result.contains("pending_turn")),
        "user line and PendingTurn must be in JSONL before tools run: {:?}",
        ui.tool_results
    );
    let loaded = JsonlSession::load(&session_path).unwrap();
    let user_lines = loaded
        .messages
        .iter()
        .filter(|message| message.role == hi_ai::Role::User)
        .filter(|message| message.text() == "unique-prompt-xyz")
        .count();
    assert_eq!(user_lines, 1, "finish must not append a second user line");
    assert!(loaded.pending_turn.is_none());
}

#[tokio::test]
async fn resume_incomplete_turn_does_not_duplicate_user_line() {
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("session.jsonl");
    let mut session = JsonlSession::create(&session_path).unwrap();
    session
        .record_turn_start(
            &Message::user("fix the parser"),
            &crate::PendingTurn {
                turn_index: 1,
                started_unix_ms: 1,
                pre_checkpoint: None,
            },
        )
        .unwrap();
    drop(session);

    let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
        text_chunk("done"),
        usage_chunk(4, 2),
    ])]) else {
        return;
    };
    let state = dir.path().join(".hi");
    let runner = ProcessRunner::new_with_policy(dir.path(), SandboxPolicy::Off).expect("runner");
    let tools = ToolHost::new_with_runner(dir.path().to_path_buf(), state.clone(), runner).unwrap();
    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.base_url = server.url.clone();
    config.state_root = state;
    config.session_path = Some(session_path.clone());
    let mut harness = Harness::new_with_tools(config, tools).unwrap();
    let loaded = JsonlSession::load(&session_path).unwrap();
    harness.apply_loaded_session(loaded);
    assert!(harness.pending_turn().is_some());
    let mut ui = TestUi::default();
    let outcome = harness
        .resume_incomplete_turn(&mut ui, TurnCancellation::new())
        .await
        .unwrap()
        .expect("pending turn should resume");
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    let loaded = JsonlSession::load(&session_path).unwrap();
    let user_lines = loaded
        .messages
        .iter()
        .filter(|message| message.role == hi_ai::Role::User)
        .filter(|message| message.text() == "fix the parser")
        .count();
    assert_eq!(user_lines, 1, "resume must not push a second user Message");
    assert!(loaded.pending_turn.is_none());
}

#[tokio::test]
async fn retry_after_resume_does_not_duplicate_user_line() {
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("session.jsonl");
    let mut session = JsonlSession::create(&session_path).unwrap();
    session
        .record_turn_start(
            &Message::user("fix the parser"),
            &crate::PendingTurn {
                turn_index: 1,
                started_unix_ms: 1,
                pre_checkpoint: None,
            },
        )
        .unwrap();
    drop(session);

    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![text_chunk("done"), usage_chunk(4, 2)]),
        Scripted::Sse(vec![text_chunk("retried"), usage_chunk(5, 2)]),
    ]) else {
        return;
    };
    let state = dir.path().join(".hi");
    let runner = ProcessRunner::new_with_policy(dir.path(), SandboxPolicy::Off).expect("runner");
    let tools = ToolHost::new_with_runner(dir.path().to_path_buf(), state.clone(), runner).unwrap();
    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.base_url = server.url.clone();
    config.state_root = state;
    config.session_path = Some(session_path.clone());
    let mut harness = Harness::new_with_tools(config, tools).unwrap();
    let loaded = JsonlSession::load(&session_path).unwrap();
    harness.apply_loaded_session(loaded);
    let last_turn_start = match harness.messages().last() {
        Some(message) if message.role == hi_ai::Role::User => {
            harness.messages().len().saturating_sub(1)
        }
        _ => harness.messages().len(),
    };
    let mut ui = TestUi::default();
    harness
        .resume_incomplete_turn(&mut ui, TurnCancellation::new())
        .await
        .unwrap()
        .expect("pending turn should resume");
    harness.truncate_messages(last_turn_start);
    harness
        .run_turn_cancellable("fix the parser", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    let loaded = JsonlSession::load(&session_path).unwrap();
    let user_lines = loaded
        .messages
        .iter()
        .filter(|message| message.role == hi_ai::Role::User)
        .filter(|message| message.text() == "fix the parser")
        .count();
    assert_eq!(
        user_lines, 1,
        "/retry after resume must not keep then re-push the user line"
    );
}

#[test]
fn resume_requires_last_message_to_be_the_pending_user_line() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.state_root = dir.path().join(".hi");
    let mut harness = Harness::new(config).unwrap();
    harness.apply_loaded_session(LoadedSession {
        messages: vec![Message::user("fix the parser")],
        pending_turn: Some(crate::PendingTurn {
            turn_index: 1,
            started_unix_ms: 1,
            pre_checkpoint: None,
        }),
        ..LoadedSession::default()
    });
    assert!(harness.can_resume_incomplete(None));
    assert!(harness.can_resume_incomplete(Some("fix the parser")));
    assert!(!harness.can_resume_incomplete(Some("other prompt")));
    harness.apply_loaded_session(LoadedSession {
        messages: vec![
            Message::user("fix the parser"),
            Message::assistant(vec![hi_ai::Content::Text("partial".into())]),
        ],
        pending_turn: Some(crate::PendingTurn {
            turn_index: 1,
            started_unix_ms: 1,
            pre_checkpoint: None,
        }),
        ..LoadedSession::default()
    });
    assert!(
        !harness.can_resume_incomplete(Some("fix the parser")),
        "compact dropping the trailing user line must fail closed"
    );
}

#[tokio::test]
async fn resume_incomplete_turn_without_pending_is_none() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = test_harness("http://127.0.0.1:1", dir.path().to_path_buf());
    let mut ui = TestUi::default();
    assert!(
        harness
            .resume_incomplete_turn(&mut ui, TurnCancellation::new())
            .await
            .unwrap()
            .is_none()
    );
}

#[test]
fn turn_intent_mode_is_recorded_on_begin_persisted_turn() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.state_root = dir.path().join(".hi");
    let mut harness = Harness::new(config).unwrap();
    harness.set_turn_intent_mode(true, true);
    assert_eq!(harness.turn_intent_mode(), (true, true));
    harness.messages.push(Message::user("fix the parser"));
    let _ = harness.begin_persisted_turn("fix the parser", None);
    assert_eq!(harness.turn_intent_mode(), (true, true));
}

#[tokio::test]
async fn heartbeat_seq_advances_during_run_turn() {
    let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
        text_chunk("all good"),
        usage_chunk(4, 2),
    ])]) else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let hb = dir.path().join("heartbeat.json");
    let publisher = hi_liveness::Publisher::new();
    let _writer = hi_liveness::spawn(
        hi_liveness::WriterConfig {
            heartbeat_path: hb.clone(),
            events_path: None,
            instance: "turn-test".into(),
            generation: 0,
            workspace: dir.path().display().to_string(),
            session_path: None,
            period: std::time::Duration::from_millis(20),
        },
        publisher.clone(),
    )
    .unwrap();
    let state = dir.path().join(".hi");
    let runner = ProcessRunner::new_with_policy(dir.path(), SandboxPolicy::Off).expect("runner");
    let tools = ToolHost::new_with_runner(dir.path().to_path_buf(), state.clone(), runner).unwrap();
    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.base_url = server.url.clone();
    config.state_root = state;
    config.liveness = Some(publisher);
    let mut harness = Harness::new_with_tools(config, tools).unwrap();
    let mut ui = TestUi::default();
    harness
        .run_turn_cancellable("hello", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut beat = None;
    while std::time::Instant::now() < deadline {
        if let Ok(parsed) = hi_liveness::read_heartbeat(&hb)
            && parsed.seq >= 1
        {
            beat = Some(parsed);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(15)).await;
    }
    let beat = beat.expect("heartbeat seq should advance during run_turn");
    assert!(beat.seq >= 1);
    assert_eq!(beat.pid, std::process::id());
    assert!(beat.last_progress_unix_ms > 0);
}

#[test]
fn doctor_sentinel_line_reports_generation() {
    assert_eq!(
        super::sentinel_doctor_line_from(true, 0),
        "sentinel: on (generation 0)"
    );
    assert_eq!(super::sentinel_doctor_line_from(false, 9), "sentinel: off");
}
