use super::*;
use crate::pipe::test_support::{
    MockPipe, Scripted, text_chunk, tool_chunk, usage_chunk, usage_chunk_with_reason,
};
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
async fn supervisor_fences_dispatch_and_waits_for_checkpoint_before_another_call() {
    struct FencedUi {
        active: bool,
        checkpoints: usize,
    }
    impl Ui for FencedUi {
        fn dispatch_allowed(&self) -> bool {
            self.active
        }
        fn mutation_batch_complete(
            &mut self,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + '_>> {
            self.checkpoints += 1;
            Box::pin(async { false })
        }
        fn assistant_text(&mut self, _: &str) {}
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str, _: &str) {}
        fn status(&mut self, _: &str) {}
        fn turn_end(&mut self, _: &str) {}
        fn turn_error(&mut self, _: &str, _: &str, _: &str) {}
    }
    let args = serde_json::json!({"path":"receipt.txt","content":"completed mutation"}).to_string();
    let server = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "mutation", "write", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("This generation must not run"),
            usage_chunk(4, 1),
        ]),
    ])
    .expect("local fixture listener is required");
    let dir = tempfile::tempdir().unwrap();
    let mut harness = test_harness(&server.url, dir.path().to_owned());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = FencedUi {
        active: false,
        checkpoints: 0,
    };
    let outcome = harness
        .run_turn_cancellable("do not dispatch", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Cancelled);
    assert!(server.bodies.lock().unwrap().is_empty());
    ui.active = true;
    let outcome = harness
        .run_turn_cancellable("write a receipt", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(
        fs::read_to_string(dir.path().join("receipt.txt")).unwrap(),
        "completed mutation"
    );
    assert_eq!(ui.checkpoints, 1);
    assert_eq!(outcome.stop_reason, TurnStopReason::Cancelled);
    assert_eq!(
        server.bodies.lock().unwrap().len(),
        1,
        "checkpoint rejection cannot trigger another generation"
    );
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
async fn truncated_output_continues_then_completes() {
    let ramble = "Let me look at handle_ws. ".repeat(80);
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            text_chunk(&ramble),
            usage_chunk_with_reason(20, 400, "length"),
        ]),
        Scripted::Sse(vec![
            text_chunk("The review is complete. No remaining issues."),
            usage_chunk(24, 12),
        ]),
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
    let text = ui.texts.join("");
    assert!(
        text.contains("The review is complete"),
        "continuation must produce a verdict, got {text}"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("truncated")),
        "status should mention truncation, got {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn truncated_output_errors_after_two_continuations() {
    let ramble = "Let me look at handle_ws. ".repeat(40);
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            text_chunk(&ramble),
            usage_chunk_with_reason(20, 400, "length"),
        ]),
        Scripted::Sse(vec![
            text_chunk(&ramble),
            usage_chunk_with_reason(22, 400, "length"),
        ]),
        Scripted::Sse(vec![
            text_chunk(&ramble),
            usage_chunk_with_reason(24, 400, "length"),
        ]),
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
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.errors.iter().any(|(kind, _)| kind == "truncated"),
        "truncated dump must not complete the turn, got {:?}",
        ui.errors
    );
}

#[tokio::test]
async fn promised_fix_without_tools_continues_then_edits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("main.rs");
    fs::write(&path, "fn main() {}\n").unwrap();
    let args = serde_json::json!({
        "path": "main.rs",
        "old_string": "fn main() {}",
        "new_string": "fn main() { println!(\"hi\"); }"
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            text_chunk("I found a planted bug. Let me fix main.rs."),
            usage_chunk(8, 20),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e", "edit", &args),
            usage_chunk(10, 8),
        ]),
        Scripted::Sse(vec![text_chunk("fixed"), usage_chunk(12, 4)]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(14, 6),
        ]),
        Scripted::Sse(vec![text_chunk("tests passed"), usage_chunk(16, 4)]),
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
    assert!(
        fs::read_to_string(&path).unwrap().contains("println!"),
        "continuation must apply the promised edit"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("without a tool call")),
        "status should mention the withheld edit, got {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn circular_review_dump_continues_without_persisting_the_ramble() {
    let mut dump = String::from("I've reviewed the codebase. Issues found:\n\n");
    for i in 1..=40 {
        dump.push_str(&format!(
            "### {i}. **`src/ws.rs` — `handle_ws` doesn't check that the ticket is not empty**\nAlready checked. Fine.\n\n"
        ));
    }
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![text_chunk(&dump), usage_chunk(20, 400)]),
        Scripted::Sse(vec![
            text_chunk("Two real bugs in ws.rs. No other changes."),
            usage_chunk(24, 12),
        ]),
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
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("repeating review dump")),
        "status should mention the dump, got {:?}",
        ui.statuses
    );
    let persisted = harness
        .messages()
        .iter()
        .map(|message| message.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !persisted.contains("doesn't check that the ticket is not empty"),
        "circular dump must not stay in history:\n{persisted}"
    );
    assert!(
        persisted.contains("omitted a truncated repeating review list"),
        "stub should replace the dump:\n{persisted}"
    );
}

#[tokio::test]
async fn claimed_fixes_without_edits_continues_then_edits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ws.rs");
    fs::write(&path, "fn handle() {}\n").unwrap();
    let args = serde_json::json!({
        "path": "ws.rs",
        "old_string": "fn handle() {}",
        "new_string": "fn handle() { /* fixed */ }"
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            text_chunk("Fixes applied\n\nI've fixed the inverted DM storage in ws.rs."),
            usage_chunk(8, 20),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e", "edit", &args),
            usage_chunk(10, 8),
        ]),
        Scripted::Sse(vec![text_chunk("done"), usage_chunk(12, 4)]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(14, 6),
        ]),
        Scripted::Sse(vec![text_chunk("tests passed"), usage_chunk(16, 4)]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("fix all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        fs::read_to_string(&path).unwrap().contains("fixed"),
        "claimed-fix continuation must apply the edit"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("no file changes")),
        "status should mention the false claim, got {:?}",
        ui.statuses
    );
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
    assert!(
        ui.tool_results.iter().any(|(_, result)| {
            result.contains("addition") && (result.contains('+') || result.contains('\u{1b}'))
        }),
        "UI must get the colored edit preview, not just 'Wrote N bytes': {:?}",
        ui.tool_results
    );
}

#[tokio::test]
async fn long_tool_loops_are_not_round_capped() {
    let dir = tempfile::tempdir().unwrap();
    let mut scripts = Vec::new();
    for i in 0..70 {
        let args = serde_json::json!({
            "path": format!("p{i}.txt"),
            "content": format!("n{i}")
        })
        .to_string();
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_{i}"), "write", &args),
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
            .filter(|(name, _)| name == "write")
            .count()
            >= 70,
        "expected 70 write rounds, got {:?}",
        ui.tool_calls.len()
    );
}

#[tokio::test]
async fn identical_tool_storm_stops_the_turn() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"command": "echo storm"}).to_string();
    let mut scripts = Vec::new();
    for i in 0..hi_liveness::IDENTICAL_TOOL_CONSECUTIVE {
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_s{i}"), "bash", &args),
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
async fn tool_storm_fills_remaining_parallel_calls() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"command": "echo storm"}).to_string();
    let mut scripts = Vec::new();
    for i in 0..hi_liveness::IDENTICAL_TOOL_CONSECUTIVE.saturating_sub(1) {
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_s{i}"), "bash", &args),
            usage_chunk(1, 1),
        ]));
    }
    scripts.push(Scripted::Sse(vec![
        tool_chunk(0, "call_storm", "bash", &args),
        tool_chunk(1, "call_rest", "bash", &args),
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
    let calls: Vec<&str> = harness
        .messages()
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|block| match block {
            hi_ai::Content::ToolCall { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    let results: Vec<&str> = harness
        .messages()
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|block| match block {
            hi_ai::Content::ToolResult { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        calls.contains(&"call_rest"),
        "parallel leftover must stay on the assistant: {calls:?}"
    );
    assert!(
        results.contains(&"call_rest"),
        "storm must not leave unmatched tool_calls: {results:?}"
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
    // Unverified-fix empty + two empty-after-tools continuations, each with
    // Pipe's four empty-stream retries, then the empty that errors.
    for _ in 0..4 {
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
        "empty-after-tools is recorded for telemetry"
    );
    assert!(
        !hi_liveness::AUTO_REPAIR_SET
            .contains(&hi_liveness::InvariantCode::EmptyAssistantAfterTools),
        "Sentinel must not kill the live child for an empty model stop"
    );
    assert!(
        harness.messages().iter().all(|message| {
            message.role != hi_ai::Role::Assistant || !message.content.is_empty()
        }),
        "empty assistant stop is not persisted"
    );
}

#[tokio::test]
async fn empty_stop_after_tools_counts_across_inspect_rounds() {
    let dir = tempfile::tempdir().unwrap();
    let first = serde_json::json!({"path": "a"}).to_string();
    let second = serde_json::json!({"path": "b"}).to_string();
    let mut scripts = vec![Scripted::Sse(vec![
        tool_chunk(0, "call_a", "list", &first),
        usage_chunk(8, 4),
    ])];
    scripts.extend(empty_stream_retries());
    scripts.push(Scripted::Sse(vec![
        tool_chunk(0, "call_b", "list", &second),
        usage_chunk(8, 4),
    ]));
    scripts.extend(empty_stream_retries());
    scripts.extend(empty_stream_retries());
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("how can we improve this", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.errors.iter().any(|(kind, _)| kind == "empty_stop"),
        "interleaved inspects must not reset empty-after-tools, got {:?}",
        ui.errors
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
    scripts.push(Scripted::Sse(vec![
        text_chunk("cargo test passed; no changes needed."),
        usage_chunk(14, 3),
    ]));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("how can we improve this", &mut ui, TurnCancellation::new())
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
async fn inspect_only_review_and_fix_continues_to_run_tests() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_l", "list", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("I reviewed the tree. No changes were necessary."),
            usage_chunk(12, 8),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(14, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("cargo test passed; nothing to fix."),
            usage_chunk(16, 4),
        ]),
    ]) else {
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
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for tests")),
        "status should mention the inspect-only continue, got {:?}",
        ui.statuses
    );
    assert!(
        ui.tool_calls
            .iter()
            .any(|(name, args)| name == "bash" && args.contains("cargo test")),
        "continuation must run cargo test, got {:?}",
        ui.tool_calls
    );
    assert!(ui.texts.join("").contains("nothing to fix"));
}

#[tokio::test]
async fn inspect_repeat_stop_on_review_and_fix_continues_to_run_tests() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_l1", "list", &args),
            tool_chunk(1, "call_l2", "list", &args),
            tool_chunk(2, "call_l3", "list", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(14, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("cargo test passed; nothing to fix."),
            usage_chunk(16, 4),
        ]),
    ]) else {
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
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for tests")),
        "inspect-repeat stop on a fix prompt must continue to tests, got {:?}",
        ui.statuses
    );
    assert!(
        ui.tool_calls
            .iter()
            .any(|(name, args)| name == "bash" && args.contains("cargo test")),
        "continuation must run cargo test, got {:?}",
        ui.tool_calls
    );
    assert!(ui.texts.join("").contains("nothing to fix"));
}

#[tokio::test]
async fn typesafe_verify_nudge_asks_for_tests_after_first_inspect_round() {
    let dir = tempfile::tempdir().unwrap();
    let grep = serde_json::json!({"pattern": "todo", "path": "."}).to_string();
    let grep2 = serde_json::json!({"pattern": "fixme", "path": "."}).to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_g", "grep", &grep),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_g2", "grep", &grep2),
            usage_chunk(10, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(14, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("cargo test passed; nothing to fix."),
            usage_chunk(16, 4),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    harness.set_next_action_override(NextAction::Verify);
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
        ui.statuses
            .iter()
            .any(|status| status.contains("typesafe next-action: verify")),
        "forced TypeSafe verify should flavor the stall-budget demand, got {:?}",
        ui.statuses
    );
    assert!(
        ui.tool_calls
            .iter()
            .any(|(name, args)| name == "bash" && args.contains("cargo test")),
        "continuation must run cargo test, got {:?}",
        ui.tool_calls
    );
    assert!(ui.texts.join("").contains("nothing to fix"));
}

#[tokio::test]
async fn typesafe_cannot_reroute_unstarted_plan_to_tests() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("main.rs");
    fs::write(&path, "fn main() {}\n").unwrap();
    let plan = serde_json::json!({
        "steps": [
            {"title": "Add /metrics endpoint", "status": "active"},
            {"title": "Remove dead code", "status": "pending"}
        ]
    })
    .to_string();
    let list = serde_json::json!({"path": "."}).to_string();
    let edit = serde_json::json!({
        "path": "main.rs",
        "old_string": "fn main() {}",
        "new_string": "fn main() { println!(\"metrics\"); }"
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let done = serde_json::json!({
        "steps": [
            {"title": "Add /metrics endpoint", "status": "done"},
            {"title": "Remove dead code", "status": "done"}
        ]
    })
    .to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_p", "update_plan", &plan),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_l", "list", &list),
            usage_chunk(10, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_r", "read", r#"{"path":"main.rs"}"#),
            usage_chunk(11, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e", "edit", &edit),
            usage_chunk(12, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("added the metrics println"),
            usage_chunk(16, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(18, 6),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_pd", "update_plan", &done),
            usage_chunk(19, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("cargo test passed after the edit"),
            usage_chunk(20, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("should not be requested"),
            usage_chunk(1, 1),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    harness.set_next_action_override(NextAction::Verify);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("lets build all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for an edit")),
        "an unstarted plan must keep the edit demand, got {:?}",
        ui.statuses
    );
    let edit_at = ui
        .statuses
        .iter()
        .position(|status| status.contains("asking for an edit"));
    let verify_at = ui
        .statuses
        .iter()
        .position(|status| status.contains("typesafe next-action: verify"));
    assert!(
        verify_at.is_none() || verify_at > edit_at,
        "TypeSafe verify must not reroute an unstarted plan onto cargo test, got {:?}",
        ui.statuses
    );
    assert!(
        fs::read_to_string(&path).unwrap().contains("metrics"),
        "continuation must edit instead of testing the current tree"
    );
    assert!(
        ui.texts
            .join("")
            .contains("cargo test passed after the edit")
    );
    assert!(
        !ui.texts.join("").contains("should not be requested"),
        "turn must stop after the edit verdict"
    );
}

#[tokio::test]
async fn typesafe_cannot_verdict_an_unstarted_plan() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("main.rs");
    fs::write(&path, "fn main() {}\n").unwrap();
    let plan = serde_json::json!({
        "steps": [
            {"title": "Add /metrics endpoint", "status": "active"},
            {"title": "Remove dead code", "status": "pending"}
        ]
    })
    .to_string();
    let list = serde_json::json!({"path": "."}).to_string();
    let edit = serde_json::json!({
        "path": "main.rs",
        "old_string": "fn main() {}",
        "new_string": "fn main() { println!(\"metrics\"); }"
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let done = serde_json::json!({
        "steps": [
            {"title": "Add /metrics endpoint", "status": "done"},
            {"title": "Remove dead code", "status": "done"}
        ]
    })
    .to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_p", "update_plan", &plan),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_l", "list", &list),
            usage_chunk(10, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_r", "read", r#"{"path":"main.rs"}"#),
            usage_chunk(11, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e", "edit", &edit),
            usage_chunk(12, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("added the metrics println"),
            usage_chunk(16, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(18, 6),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_pd", "update_plan", &done),
            usage_chunk(19, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("cargo test passed after the edit"),
            usage_chunk(20, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("should not be requested"),
            usage_chunk(1, 1),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    harness.set_next_action_override(NextAction::Verdict);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("lets build all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for an edit")),
        "an unstarted plan must keep the edit demand, got {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("typesafe next-action: verdict")
                || status.contains("asking for a verdict")),
        "TypeSafe verdict must not close an unstarted plan, got {:?}",
        ui.statuses
    );
    assert!(
        fs::read_to_string(&path).unwrap().contains("metrics"),
        "continuation must edit instead of writing a cop-out"
    );
    assert!(
        !ui.texts.join("").contains("should not be requested"),
        "turn must stop after the edit verdict"
    );
}

#[tokio::test]
async fn inspect_repeat_after_cargo_test_asks_for_verdict() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let mut scripts = vec![Scripted::Sse(vec![
        tool_chunk(0, "call_t", "bash", &test_cmd),
        usage_chunk(8, 4),
    ])];
    for i in 0..2 {
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_g{i}"), "list", &args),
            usage_chunk(10, 4),
        ]));
    }
    scripts.push(Scripted::Sse(vec![
        text_chunk("make the join flow one command; skip more greps."),
        usage_chunk(16, 4),
    ]));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable(
            "how to improve this program? make it better, easier to use and faster?",
            &mut ui,
            TurnCancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for a verdict")),
        "inspect-repeat after cargo test on an improve prompt must ask for a written answer, got {:?}",
        ui.statuses
    );
    assert!(ui.texts.join("").contains("join flow"));
}

#[tokio::test]
async fn open_plan_inspect_loop_asks_for_edit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("main.rs");
    fs::write(&path, "fn main() {}\n").unwrap();
    let plan = serde_json::json!({
        "steps": [
            {"title": "Add /metrics endpoint", "status": "active"},
            {"title": "Remove dead code", "status": "pending"}
        ]
    })
    .to_string();
    let list = serde_json::json!({"path": "."}).to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let edit = serde_json::json!({
        "path": "main.rs",
        "old_string": "fn main() {}",
        "new_string": "fn main() { println!(\"metrics\"); }"
    })
    .to_string();
    let done = serde_json::json!({
        "steps": [
            {"title": "Add /metrics endpoint", "status": "done"},
            {"title": "Remove dead code", "status": "done"}
        ]
    })
    .to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_p", "update_plan", &plan),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_l", "list", &list),
            usage_chunk(10, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_r", "read", r#"{"path":"main.rs"}"#),
            usage_chunk(11, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e", "edit", &edit),
            usage_chunk(12, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("added the metrics println"),
            usage_chunk(16, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(18, 6),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_pd", "update_plan", &done),
            usage_chunk(19, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("cargo test passed after the edit"),
            usage_chunk(20, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("should not be requested"),
            usage_chunk(1, 1),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("do all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for an edit")),
        "an open plan plus inspect must ask for an edit, got {:?}",
        ui.statuses
    );
    assert!(
        fs::read_to_string(&path).unwrap().contains("metrics"),
        "continuation must edit instead of closing the plan"
    );
    assert!(
        ui.texts
            .join("")
            .contains("cargo test passed after the edit")
    );
    assert!(
        !ui.texts.join("").contains("should not be requested"),
        "turn must stop after the edit verdict"
    );
}

#[tokio::test]
async fn open_plan_empty_stop_asks_for_edit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("main.rs");
    fs::write(&path, "fn main() {}\n").unwrap();
    let plan = serde_json::json!({
        "steps": [
            {"title": "Add /metrics endpoint", "status": "active"},
            {"title": "Run clippy and tests", "status": "pending"}
        ]
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let edit = serde_json::json!({
        "path": "main.rs",
        "old_string": "fn main() {}",
        "new_string": "fn main() { println!(\"metrics\"); }"
    })
    .to_string();
    let done = serde_json::json!({
        "steps": [
            {"title": "Add /metrics endpoint", "status": "done"},
            {"title": "Run clippy and tests", "status": "done"}
        ]
    })
    .to_string();
    let mut scripts = vec![Scripted::Sse(vec![
        tool_chunk(0, "call_p", "update_plan", &plan),
        usage_chunk(8, 4),
    ])];
    scripts.extend(empty_stream_retries());
    scripts.push(Scripted::Sse(vec![
        tool_chunk(0, "call_e", "edit", &edit),
        usage_chunk(12, 6),
    ]));
    scripts.push(Scripted::Sse(vec![
        text_chunk("added metrics and stopped inspecting"),
        usage_chunk(16, 4),
    ]));
    scripts.push(Scripted::Sse(vec![
        tool_chunk(0, "call_t", "bash", &test_cmd),
        usage_chunk(18, 6),
    ]));
    scripts.push(Scripted::Sse(vec![
        tool_chunk(0, "call_pd", "update_plan", &done),
        usage_chunk(19, 4),
    ]));
    scripts.push(Scripted::Sse(vec![
        text_chunk("cargo test passed after the edit"),
        usage_chunk(20, 4),
    ]));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("do all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for an edit")),
        "empty stop after update_plan must ask for an edit, got {:?}",
        ui.statuses
    );
    assert!(
        fs::read_to_string(&path).unwrap().contains("metrics"),
        "empty stop with an open plan must not complete without an edit"
    );
}

#[tokio::test]
async fn implement_prompt_inspect_loop_asks_for_edit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("main.rs");
    fs::write(&path, "fn main() {}\n").unwrap();
    let list = serde_json::json!({"path": "."}).to_string();
    let grep = serde_json::json!({"pattern": "metrics", "path": "main.rs"}).to_string();
    let edit = serde_json::json!({
        "path": "main.rs",
        "old_string": "fn main() {}",
        "new_string": "fn main() { println!(\"metrics\"); }"
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_l", "list", &list),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_g", "grep", &grep),
            usage_chunk(10, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e", "edit", &edit),
            usage_chunk(12, 6),
        ]),
        Scripted::Sse(vec![text_chunk("added metrics"), usage_chunk(14, 4)]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(16, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("cargo test passed after the edit"),
            usage_chunk(18, 4),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("do all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for an edit")),
        "implement prompt inspect loop must ask for an edit, got {:?}",
        ui.statuses
    );
    assert!(fs::read_to_string(&path).unwrap().contains("metrics"));
    assert!(
        ui.texts
            .join("")
            .contains("cargo test passed after the edit")
    );
}

#[tokio::test]
async fn implement_prompt_inspect_budget_stops_without_auto_repair() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}\n").unwrap();
    let mut scripts = Vec::new();
    for i in 0..crate::completion::STALL_ROUNDS_BEFORE_ERROR {
        let args = serde_json::json!({"path": format!("p{i}")}).to_string();
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_{i}"), "list", &args),
            usage_chunk(8, 4),
        ]));
    }
    scripts.push(Scripted::Sse(vec![
        text_chunk("should not be requested after inspect budget"),
        usage_chunk(1, 1),
    ]));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("build all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.errors.iter().any(|(kind, _)| kind == "inspect_stop"),
        "inspect budget must end the turn, got {:?}",
        ui.errors
    );
    assert!(
        harness.liveness().snapshot().invariant.is_none(),
        "inspect budget is not a harness crash: {:?}",
        harness.liveness().snapshot().invariant
    );
    assert!(
        !ui.texts
            .join("")
            .contains("should not be requested after inspect budget"),
        "must not keep inspecting after the budget"
    );
}

#[tokio::test]
async fn plan_then_verify_then_offset_reads_never_complete_silently() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("web.rs"), "fn handler() {}\n").unwrap();
    let plan = serde_json::json!({
        "steps": [
            {"title": "Add /metrics endpoint", "status": "active"},
            {"title": "Remove dead code", "status": "pending"}
        ]
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let mut scripts = vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_p", "update_plan", &plan),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(10, 4),
        ]),
    ];
    // Opening update_plan is not a stall. cargo test holds the count.
    // The first offset read is a unique path; later offsets are pagination.
    // Cap + last-chance + one more page ends the turn.
    let reads = crate::completion::STALL_ROUNDS_BEFORE_ERROR.saturating_add(2);
    for i in 0..reads {
        let read = serde_json::json!({"path": "web.rs", "offset": i * 40}).to_string();
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_r{i}"), "read", &read),
            usage_chunk(12, 4),
        ]));
    }
    scripts.push(Scripted::Sse(vec![
        text_chunk("should not silently complete"),
        usage_chunk(1, 1),
    ]));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("do all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.errors
            .iter()
            .any(|(kind, _)| kind == "plan_stall" || kind == "inspect_stop"),
        "mid-plan inspect after cargo test must not complete, got {:?}",
        ui.errors
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for an edit")),
        "stall budget must demand an edit, got {:?}",
        ui.statuses
    );
    assert!(
        !ui.texts.join("").contains("should not silently complete"),
        "must not accept a silent or cop-out close after the stall"
    );
}

#[tokio::test]
async fn live_chat_paginated_reads_still_reach_an_edit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("server.rs");
    let body: String = (0..120)
        .map(|i| format!("line {i} of the handler\n"))
        .collect();
    fs::write(&path, &body).unwrap();
    let plan = serde_json::json!({
        "steps": [
            {"title": "FTS5 search", "status": "active"},
            {"title": "REST history API", "status": "pending"}
        ]
    })
    .to_string();
    let edit = serde_json::json!({
        "path": "server.rs",
        "old_string": "line 0 of the handler",
        "new_string": "line 0 of the handler // metrics"
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let done = serde_json::json!({
        "steps": [
            {"title": "FTS5 search", "status": "done"},
            {"title": "REST history API", "status": "done"}
        ]
    })
    .to_string();
    let mut scripts = vec![Scripted::Sse(vec![
        tool_chunk(0, "call_p", "update_plan", &plan),
        usage_chunk(8, 4),
    ])];
    // Nine offset reads of one file: unique-file stall used to plan_stall on
    // the last page (live ~/chat paging server.rs after "build all of that").
    for i in 0..9 {
        let read = serde_json::json!({"path": "server.rs", "offset": i * 12}).to_string();
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_r{i}"), "read", &read),
            usage_chunk(12, 4),
        ]));
    }
    scripts.push(Scripted::Sse(vec![
        tool_chunk(0, "call_e", "edit", &edit),
        usage_chunk(14, 6),
    ]));
    scripts.push(Scripted::Sse(vec![
        text_chunk("patched server.rs"),
        usage_chunk(16, 4),
    ]));
    scripts.push(Scripted::Sse(vec![
        tool_chunk(0, "call_t", "bash", &test_cmd),
        usage_chunk(18, 6),
    ]));
    scripts.push(Scripted::Sse(vec![
        tool_chunk(0, "call_pd", "update_plan", &done),
        usage_chunk(19, 4),
    ]));
    scripts.push(Scripted::Sse(vec![
        text_chunk("cargo test passed after the edit"),
        usage_chunk(20, 4),
    ]));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("build all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        !ui.errors.iter().any(|(kind, _)| kind == "plan_stall"),
        "paging one file must not plan_stall before the edit, got {:?}",
        ui.errors
    );
    assert!(
        fs::read_to_string(&path).unwrap().contains("metrics"),
        "the model must be allowed to edit after paginated reads"
    );
}

#[tokio::test]
async fn partial_plan_cop_out_after_edit_does_not_complete() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.rs");
    fs::write(&path, "fn pool() {}\n").unwrap();
    let plan = serde_json::json!({
        "steps": [
            {"title": "Connection pool", "status": "active"},
            {"title": "Forward HISTORY pagination", "status": "pending"}
        ]
    })
    .to_string();
    let mid = serde_json::json!({
        "steps": [
            {"title": "Connection pool", "status": "done"},
            {"title": "Forward HISTORY pagination", "status": "active"}
        ]
    })
    .to_string();
    let done = serde_json::json!({
        "steps": [
            {"title": "Connection pool", "status": "done"},
            {"title": "Forward HISTORY pagination", "status": "done"}
        ]
    })
    .to_string();
    let edit1 = serde_json::json!({
        "path": "db.rs",
        "old_string": "fn pool() {}",
        "new_string": "fn pool() { /* pooled */ }"
    })
    .to_string();
    let edit2 = serde_json::json!({
        "path": "db.rs",
        "old_string": "fn pool() { /* pooled */ }",
        "new_string": "fn pool() { /* pooled+history */ }"
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let grep = serde_json::json!({"pattern": "HISTORY", "path": "db.rs"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_p", "update_plan", &plan),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e1", "edit", &edit1),
            usage_chunk(12, 6),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(14, 6),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_pm", "update_plan", &mid),
            usage_chunk(16, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_g", "grep", &grep),
            usage_chunk(18, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk(
                "I'll stop inspecting and give you the answer. Remaining: HISTORY pagination.",
            ),
            usage_chunk(20, 8),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e2", "edit", &edit2),
            usage_chunk(22, 6),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t2", "bash", &test_cmd),
            usage_chunk(24, 6),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_pd", "update_plan", &done),
            usage_chunk(26, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("wired HISTORY pagination; cargo test passed"),
            usage_chunk(28, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("should not complete on the cop-out"),
            usage_chunk(1, 1),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("build all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.texts.join("").contains("I'll stop inspecting"),
        "cop-out must be visible, got {:?}",
        ui.texts
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for an edit")),
        "mid-plan cop-out after a partial edit must demand the rest, got {:?}",
        ui.statuses
    );
    assert!(
        fs::read_to_string(&path).unwrap().contains("history"),
        "continuation must finish remaining plan steps"
    );
    assert!(ui.texts.join("").contains("wired HISTORY pagination"));
    assert!(
        !ui.texts
            .join("")
            .contains("should not complete on the cop-out"),
        "must not stop on the cop-out"
    );
}

#[tokio::test]
async fn dishonest_all_done_plan_after_one_edit_does_not_complete() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.rs");
    fs::write(&path, "fn pool() {}\n").unwrap();
    let plan = serde_json::json!({
        "steps": [
            {"title": "Connection pool", "status": "active"},
            {"title": "Forward HISTORY pagination", "status": "pending"},
            {"title": "Rate limiter", "status": "pending"}
        ]
    })
    .to_string();
    let all_done = serde_json::json!({
        "steps": [
            {"title": "Connection pool", "status": "done"},
            {"title": "Forward HISTORY pagination", "status": "done"},
            {"title": "Rate limiter", "status": "done"}
        ]
    })
    .to_string();
    let edit = serde_json::json!({
        "path": "db.rs",
        "old_string": "fn pool() {}",
        "new_string": "fn pool() { /* pooled */ }"
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_p", "update_plan", &plan),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e", "edit", &edit),
            usage_chunk(12, 6),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(14, 6),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_pd", "update_plan", &all_done),
            usage_chunk(16, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("I'll stop inspecting. Remaining: HISTORY pagination, rate limiter."),
            usage_chunk(20, 8),
        ]),
        Scripted::Sse(vec![
            text_chunk("still remaining, not implementing those"),
            usage_chunk(22, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("third cop-out after a dishonest plan close"),
            usage_chunk(24, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("should not be requested after dishonest close"),
            usage_chunk(1, 1),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("build all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.errors.iter().any(|(kind, _)| kind == "plan_stall"),
        "bulk-closing leftover plan steps after one edit must plan_stall, got {:?}",
        ui.errors
    );
    assert!(
        crate::completion::plan_is_open(harness.current_plan()),
        "harness must keep unfinished steps, got {:?}",
        harness.current_plan()
    );
    assert!(
        !ui.texts
            .join("")
            .contains("should not be requested after dishonest close"),
        "must not accept a bulk plan close"
    );
}

#[tokio::test]
async fn implement_prompt_prose_after_inspect_asks_for_edit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("main.rs");
    fs::write(&path, "fn main() {}\n").unwrap();
    let list = serde_json::json!({"path": "."}).to_string();
    let edit = serde_json::json!({
        "path": "main.rs",
        "old_string": "fn main() {}",
        "new_string": "fn main() { println!(\"metrics\"); }"
    })
    .to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_l", "list", &list),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("The handlers look fine. Metrics can wait."),
            usage_chunk(10, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e", "edit", &edit),
            usage_chunk(12, 6),
        ]),
        Scripted::Sse(vec![text_chunk("added metrics"), usage_chunk(14, 4)]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(16, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("cargo test passed after the edit"),
            usage_chunk(18, 4),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("do all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for an edit")),
        "implement prompt must not accept a prose cop-out after inspect, got {:?}",
        ui.statuses
    );
    assert!(fs::read_to_string(&path).unwrap().contains("metrics"));
}

#[tokio::test]
async fn later_improve_prompt_does_not_inherit_fix_intent() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![text_chunk("No major issues."), usage_chunk(6, 4)]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_l", "list", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("Add a /metrics endpoint; skip more greps."),
            usage_chunk(12, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("should not be requested"),
            usage_chunk(1, 1),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let first = harness
        .run_turn_cancellable(
            "review for any major issues and fix",
            &mut ui,
            TurnCancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(first.stop_reason, TurnStopReason::Completed);
    let second = harness
        .run_turn_cancellable("how can we improve this", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(second.stop_reason, TurnStopReason::Completed);
    assert!(
        !ui.tool_calls
            .iter()
            .any(|(name, args)| name == "bash" && args.contains("cargo test")),
        "an improve follow-up must not inherit review-and-fix test pressure, got {:?}",
        ui.tool_calls
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("asking for tests")),
        "improve follow-up status leaked a fix continue, got {:?}",
        ui.statuses
    );
    assert!(ui.texts.join("").contains("/metrics endpoint"));
    assert!(!ui.texts.join("").contains("should not be requested"));
}

#[tokio::test]
async fn post_edit_review_and_fix_reruns_tests() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lib.rs");
    fs::write(&path, "fn add() { assert_eq!(2 + 2, 5); }\n").unwrap();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let edit = serde_json::json!({
        "path": "lib.rs",
        "old_string": "assert_eq!(2 + 2, 5)",
        "new_string": "assert_eq!(2 + 2, 4)"
    })
    .to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_t0", "bash", &test_cmd),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_e", "edit", &edit),
            usage_chunk(10, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("I applied the assertion fix."),
            usage_chunk(12, 8),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t1", "bash", &test_cmd),
            usage_chunk(14, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("cargo test passed after the edit."),
            usage_chunk(16, 4),
        ]),
    ]) else {
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
        fs::read_to_string(&path).unwrap().contains("2 + 2, 4"),
        "planted assertion must be patched"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for tests")),
        "a pre-edit cargo test must not skip post-edit verification, got {:?}",
        ui.statuses
    );
    let cargo_tests = ui
        .tool_calls
        .iter()
        .filter(|(name, args)| name == "bash" && args.contains("cargo test"))
        .count();
    assert!(
        cargo_tests >= 2,
        "must re-run cargo test after the edit, got {:?}",
        ui.tool_calls
    );
    assert!(ui.texts.join("").contains("passed after the edit"));
}

#[tokio::test]
async fn post_edit_inspect_loop_asks_for_tests() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lib.rs");
    fs::write(&path, "fn add() {}\n").unwrap();
    let edit = serde_json::json!({
        "path": "lib.rs",
        "old_string": "fn add() {}",
        "new_string": "fn add() { let _ = 1; }"
    })
    .to_string();
    let grep_a = serde_json::json!({"pattern": "handle_ws", "path": "lib.rs"}).to_string();
    let grep_b = serde_json::json!({"pattern": "MAX_PASSWORD", "path": "lib.rs"}).to_string();
    let test_cmd = serde_json::json!({"command": "cargo test --offline --quiet"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_e", "edit", &edit),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_g1", "grep", &grep_a),
            usage_chunk(10, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_g2", "grep", &grep_b),
            usage_chunk(12, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_t", "bash", &test_cmd),
            usage_chunk(14, 6),
        ]),
        Scripted::Sse(vec![
            text_chunk("cargo test passed; patch is verified."),
            usage_chunk(16, 4),
        ]),
    ]) else {
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
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for tests")),
        "post-edit inspect rounds must ask for tests, got {:?}",
        ui.statuses
    );
    assert!(
        ui.tool_calls
            .iter()
            .any(|(name, args)| name == "bash" && args.contains("cargo test")),
        "continuation must run cargo test, got {:?}",
        ui.tool_calls
    );
    assert!(ui.texts.join("").contains("patch is verified"));
}

#[tokio::test]
async fn inspect_only_without_fix_prompt_does_not_force_tests() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_l", "list", &args),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![text_chunk("listing looks fine"), usage_chunk(12, 3)]),
        Scripted::Sse(vec![
            text_chunk("should not be requested"),
            usage_chunk(1, 1),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("summarize the tree", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(ui.texts.join("").contains("listing looks fine"));
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.contains("asking for tests")),
        "non-fix prompts must not force a test run, got {:?}",
        ui.statuses
    );
    assert!(
        !ui.texts.join("").contains("should not be requested"),
        "third script must not run"
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
    scripts.push(Scripted::Sse(vec![
        text_chunk("probes failed; stopping."),
        usage_chunk(16, 4),
    ]));
    // A sixth request would consume this and fail the assertion below.
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
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.errors.iter().any(|(kind, _)| kind == "inspect_stop"),
        "probe-repeat cop-out after the verify demand must not complete, got {:?}",
        ui.errors
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for tests")),
        "probe-repeat must demand a next action, got {:?}",
        ui.statuses
    );
    assert!(ui.texts.join("").contains("probes failed"));
    let requests = server.bodies.lock().unwrap().len();
    assert_eq!(
        requests, 5,
        "four probe rounds then one demand request; got {requests}"
    );
    assert!(
        !ui.texts.join("").contains("should not be requested"),
        "fifth script must not run"
    );
}

#[tokio::test]
async fn second_inspect_refusal_stops_the_turn() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let mut scripts = Vec::new();
    for i in 0..2 {
        scripts.push(Scripted::Sse(vec![
            tool_chunk(0, &format!("call_g{i}"), "list", &args),
            usage_chunk(8, 4),
        ]));
    }
    scripts.push(Scripted::Sse(vec![
        text_chunk("here is the short plan"),
        usage_chunk(16, 4),
    ]));
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
        .run_turn_cancellable("build all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.errors.iter().any(|(kind, _)| kind == "inspect_stop"),
        "implement cop-out after the edit demand must not complete, got {:?}",
        ui.errors
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for an edit")),
        "inspect-repeat stop on an implement prompt must ask for an edit, got {:?}",
        ui.statuses
    );
    assert!(ui.texts.join("").contains("here is the short plan"));
    let requests = server.bodies.lock().unwrap().len();
    assert_eq!(
        requests, 3,
        "two inspect rounds then one edit request; got {requests}"
    );
    assert!(
        !ui.texts.join("").contains("should not be requested"),
        "fifth script must not run"
    );
}

#[tokio::test]
async fn parallel_identical_inspects_stop_without_tool_storm() {
    let dir = tempfile::tempdir().unwrap();
    let args = serde_json::json!({"path": "."}).to_string();
    let mut chunks = Vec::new();
    for i in 0..hi_liveness::IDENTICAL_TOOL_CONSECUTIVE as usize {
        chunks.push(tool_chunk(i, &format!("call_g{i}"), "list", &args));
    }
    chunks.push(usage_chunk(8, 4));
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(chunks),
        Scripted::Sse(vec![
            text_chunk("here is the short plan"),
            usage_chunk(16, 4),
        ]),
        Scripted::Sse(vec![
            text_chunk("should not be requested"),
            usage_chunk(1, 1),
        ]),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_permission_mode(PermissionMode::Always);
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable("build all of that", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.errors.iter().any(|(kind, _)| kind == "inspect_stop"),
        "implement cop-out after the edit demand must not complete, got {:?}",
        ui.errors
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for an edit")),
        "parallel inspect refusals on an implement prompt must ask for an edit, got {:?}",
        ui.statuses
    );
    assert!(ui.texts.join("").contains("here is the short plan"));
    assert!(
        !ui.errors.iter().any(|(kind, _)| kind == "tool_storm"),
        "refused inspects must not count as a tool storm, got {:?}",
        ui.errors
    );
    assert_eq!(
        harness.liveness().snapshot().invariant.map(|inv| inv.code),
        None
    );
    assert_eq!(server.bodies.lock().unwrap().len(), 2);
    assert!(
        !ui.texts.join("").contains("should not be requested"),
        "third script must not run"
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
        text_chunk("probes failed; stopping."),
        usage_chunk(16, 4),
    ]));
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
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.errors.iter().any(|(kind, _)| kind == "inspect_stop"),
        "probe-repeat cop-out after the verify demand must not complete, got {:?}",
        ui.errors
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("asking for tests")),
        "probe-repeat must demand a next action instead of a silent close, got {:?}",
        ui.statuses
    );
    assert!(ui.texts.join("").contains("probes failed"));
    let requests = server.bodies.lock().unwrap().len();
    assert_eq!(
        requests, 5,
        "four probe rounds then one demand request; got {requests} command={command}"
    );
    assert!(
        !ui.texts.join("").contains("should not be requested"),
        "sixth script must not run command={command}"
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
fn resume_requires_the_pending_user_line_even_with_mid_turn_tools() {
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
            Message::assistant(vec![
                hi_ai::Content::Text("partial".into()),
                hi_ai::Content::ToolCall {
                    id: "c1".into(),
                    name: "bash".into(),
                    arguments: "{}".into(),
                },
            ]),
            Message::tool_result("c1", "ok"),
        ],
        pending_turn: Some(crate::PendingTurn {
            turn_index: 1,
            started_unix_ms: 1,
            pre_checkpoint: None,
        }),
        ..LoadedSession::default()
    });
    assert!(
        harness.can_resume_incomplete(Some("fix the parser")),
        "mid-turn assistant/tool after the matching user must still resume"
    );
    harness.apply_loaded_session(LoadedSession {
        messages: vec![
            Message::user("fix the parser"),
            Message::assistant(vec![hi_ai::Content::ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                arguments: "{}".into(),
            }]),
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
        "assistant tool_call without a result is not a valid resume transcript"
    );
    harness.apply_loaded_session(LoadedSession {
        messages: vec![
            Message::user("old prompt"),
            Message::assistant(vec![hi_ai::Content::Text("summary".into())]),
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
        "compact dropping the in-flight user line must fail closed"
    );
}

#[tokio::test]
async fn persist_mid_turn_tool_results_before_the_next_round() {
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("session.jsonl");
    let first = serde_json::json!({"command": "echo unique-mid-turn-xyz"}).to_string();
    let second = serde_json::json!({"command": "cat session.jsonl"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_a", "bash", &first),
            usage_chunk(8, 4),
        ]),
        Scripted::Sse(vec![
            tool_chunk(0, "call_b", "bash", &second),
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
    harness
        .run_turn_cancellable("persist as you go", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    let cat = ui
        .tool_results
        .iter()
        .find(|(name, result)| name == "bash" && result.contains("pending_turn"))
        .map(|(_, result)| result.as_str())
        .expect("second bash should cat the session");
    assert!(
        cat.contains("unique-mid-turn-xyz"),
        "first tool result must be in JSONL before the next round: {cat}"
    );
}

#[tokio::test]
async fn persist_does_not_record_unmatched_tool_calls_mid_round() {
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("session.jsonl");
    let cat = serde_json::json!({"command": "cat session.jsonl"}).to_string();
    let echo = serde_json::json!({"command": "echo parallel-b"}).to_string();
    let Some(server) = MockPipe::new(vec![
        Scripted::Sse(vec![
            tool_chunk(0, "call_a", "bash", &cat),
            tool_chunk(1, "call_b", "bash", &echo),
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
    harness
        .run_turn_cancellable("parallel tools", &mut ui, TurnCancellation::new())
        .await
        .unwrap();
    let cat_out = ui
        .tool_results
        .iter()
        .find(|(name, result)| name == "bash" && result.contains("pending_turn"))
        .map(|(_, result)| result.as_str())
        .expect("first parallel bash should cat the session");
    assert!(
        !cat_out.contains("call_a"),
        "assistant tool_calls must not hit JSONL before their results: {cat_out}"
    );
}

#[tokio::test]
async fn resume_incomplete_with_persisted_tools_does_not_duplicate_user_line() {
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
    session
        .record_messages(&[
            Message::assistant(vec![
                hi_ai::Content::Text("working".into()),
                hi_ai::Content::ToolCall {
                    id: "c1".into(),
                    name: "bash".into(),
                    arguments: "{}".into(),
                },
            ]),
            Message::tool_result("c1", "ok"),
        ])
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
    let mut ui = TestUi::default();
    let outcome = harness
        .resume_incomplete_turn(&mut ui, TurnCancellation::new())
        .await
        .unwrap()
        .expect("pending turn with tools should resume");
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
fn failed_compact_rewrite_does_not_append_onto_old_session_file() {
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("session.jsonl");
    let state = dir.path().join(".hi");
    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.state_root = state;
    config.session_path = Some(session_path.clone());
    let mut harness = Harness::new(config).unwrap();
    harness.messages = vec![
        Message::user("one"),
        Message::assistant(vec![hi_ai::Content::Text("two".into())]),
        Message::user("three"),
        Message::assistant(vec![hi_ai::Content::Text("four".into())]),
    ];
    assert!(harness.persist_snapshot());
    let mut persisted_before = harness.messages.len();

    let mut tmp = session_path.as_os_str().to_os_string();
    tmp.push(".tmp");
    fs::create_dir(&tmp).unwrap();

    let compacted = vec![Message::user("one"), Message::user("summary of the rest")];
    assert!(
        !harness.replace_messages_persisting(compacted),
        "rewrite must fail while the session tmp path is a directory"
    );
    assert_eq!(harness.messages().len(), 4);
    assert_eq!(harness.messages()[3].text(), "four");

    harness.messages.push(Message::user("steer"));
    harness.persist_turn_progress(&mut persisted_before);
    let loaded = JsonlSession::load(&session_path).unwrap();
    let file: Vec<_> = loaded.messages.iter().map(|m| m.text()).collect();
    let memory: Vec<_> = harness.messages().iter().map(|m| m.text()).collect();
    assert_eq!(file, memory);
    assert_eq!(file, vec!["one", "two", "three", "four", "steer"]);
    assert_ne!(
        harness.liveness().snapshot().invariant.map(|inv| inv.code),
        Some(hi_liveness::InvariantCode::SessionAppendFailed),
        "a rolled-back rewrite must not auto-repair the live turn"
    );
}

#[test]
fn begin_turn_persist_failure_does_not_auto_repair() {
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("session.jsonl");
    let state = dir.path().join(".hi");
    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.state_root = state;
    config.session_path = Some(session_path.clone());
    let mut harness = Harness::new(config).unwrap();
    fs::remove_file(&session_path).unwrap();
    fs::create_dir(&session_path).unwrap();
    harness.messages.push(Message::user("hello"));
    let from = harness.begin_persisted_turn("hello", None);
    assert_eq!(from, 0, "cursor must stay on the unpersisted user line");
    assert!(
        harness.liveness().snapshot().invariant.is_none(),
        "a retryable turn-start fsync must not auto-repair the live turn"
    );
}

#[test]
fn persist_progress_after_failed_begin_writes_pending() {
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("session.jsonl");
    let state = dir.path().join(".hi");
    let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
    config.state_root = state;
    config.session_path = Some(session_path.clone());
    let mut harness = Harness::new(config).unwrap();
    fs::remove_file(&session_path).unwrap();
    fs::create_dir(&session_path).unwrap();
    harness.messages.push(Message::user("hello"));
    let mut from = harness.begin_persisted_turn("hello", None);
    assert_eq!(from, 0);
    fs::remove_dir(&session_path).unwrap();
    fs::write(&session_path, "").unwrap();
    harness.persist_turn_progress(&mut from);
    assert_eq!(from, 1);
    let loaded = JsonlSession::load(&session_path).unwrap();
    assert_eq!(loaded.messages.len(), 1);
    assert_eq!(loaded.messages[0].text(), "hello");
    assert!(
        loaded.pending_turn.is_some(),
        "retry append after a failed turn-start must still record PendingTurn"
    );
}

#[test]
fn doctor_sentinel_line_reports_generation() {
    assert_eq!(
        super::sentinel_doctor_line_from(true, 0),
        "sentinel: on (generation 0)"
    );
    assert_eq!(super::sentinel_doctor_line_from(false, 9), "sentinel: off");
}
