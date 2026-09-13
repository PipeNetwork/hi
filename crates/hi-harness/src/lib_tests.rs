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
    let args = serde_json::json!({"path": "."}).to_string();
    let mut scripts = Vec::new();
    for i in 0..70 {
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

#[tokio::test]
async fn empty_stop_after_tools_ends_the_turn() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_l", "list", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![usage_chunk(10, 0)]),
        Scripted::Sse(vec![usage_chunk(10, 0)]),
        Scripted::Sse(vec![usage_chunk(10, 0)]),
        Scripted::Sse(vec![usage_chunk(10, 0)]),
    ]) else {
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
    assert!(ui.texts.is_empty());
    assert!(
        harness.messages().iter().all(|message| {
            message.role != hi_ai::Role::Assistant || !message.content.is_empty()
        }),
        "empty assistant stop is not persisted"
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
