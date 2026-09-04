use super::*;
use crate::event::UiEvent;

fn drive(input: &str) -> Vec<Value> {
    let mut output = Vec::new();
    run_jsonl(std::io::Cursor::new(input), &mut output).unwrap();
    String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn jsonl_render_is_repeatable_and_uses_real_input_path() {
    let input = concat!(
        "{\"id\":1,\"command\":\"resize\",\"width\":64,\"height\":18}\n",
        "{\"id\":2,\"command\":\"paste\",\"text\":\"hello harness\"}\n",
        "{\"id\":3,\"command\":\"key\",\"key\":\"left\"}\n",
        "{\"id\":4,\"command\":\"key\",\"key\":\"right\"}\n",
        "{\"id\":5,\"command\":\"key\",\"key\":\"enter\"}\n",
        "{\"id\":6,\"command\":\"render\"}\n",
        "{\"id\":7,\"command\":\"render\"}\n",
    );
    let first = drive(input);
    let second = drive(input);
    assert_eq!(first, second);
    assert_eq!(first[4]["result"]["submitted"], "hello harness");
    assert_eq!(first[5]["result"]["digest"], first[6]["result"]["digest"]);
    assert!(
        first[5]["result"]["lines"]
            .as_array()
            .unwrap()
            .iter()
            .any(|line| line.as_str().unwrap().contains("hello harness"))
    );
}

#[test]
fn inspect_tracks_focus_and_transcript_block_lifecycle() {
    let rows = drive(concat!(
        "{\"command\":\"focus\",\"focused\":false}\n",
        "{\"command\":\"transcript\",\"event\":{\"kind\":\"reasoning\",\"text\":\"checking\"}}\n",
        "{\"command\":\"transcript\",\"event\":{\"kind\":\"text\",\"text\":\"answer\"}}\n",
        "{\"command\":\"transcript\",\"event\":{\"kind\":\"assistant_end\"}}\n",
        "{\"command\":\"transcript\",\"event\":{\"kind\":\"tool_call\",\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}\n",
        "{\"command\":\"transcript\",\"event\":{\"kind\":\"tool_result\",\"name\":\"bash\",\"result\":\"/workspace\"}}\n",
        "{\"command\":\"inspect\"}\n",
    ));
    let tree = &rows.last().unwrap()["result"]["component_tree"]["root"];
    assert_eq!(tree["focused"], false);
    let blocks = tree["children"][0]["children"].as_array().unwrap();
    assert!(
        blocks
            .iter()
            .any(|node| node["kind"] == "assistant_message")
    );
    assert!(blocks.iter().any(|node| node["kind"] == "activity_run"));
}

#[test]
fn session_events_and_snapshots_share_the_versioned_projection() {
    let rows = drive(concat!(
        "{\"id\":1,\"command\":\"session_event\",\"event\":{\"schema_version\":1,\"sequence\":1,\"kind\":{\"type\":\"message\",\"message\":{\"role\":\"User\",\"content\":[{\"Text\":\"from projection\"}]}}}}\n",
        "{\"id\":2,\"command\":\"inspect\"}\n",
    ));
    let projection = &rows[1]["result"]["session_projection"];
    assert_eq!(projection["revision"], 1);
    assert_eq!(
        projection["reducer_version"],
        hi_agent::SESSION_REDUCER_VERSION
    );
    assert!(
        projection["digest"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );
    assert!(
        rows[1]["result"]["component_tree"]["root"]["children"][0]["children"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["kind"] == "user_prompt")
    );

    let mut second = Harness::default();
    let snapshot: hi_agent::SessionProjectionSnapshot =
        serde_json::from_value(projection.clone()).unwrap();
    second
        .handle(Command::SessionSnapshot {
            snapshot: Box::new(snapshot),
        })
        .unwrap();
    assert_eq!(second.app.session_projection_snapshot().revision, 1);
    assert!(second.app.transcript_text().contains("from projection"));
}

#[test]
fn exact_base_projection_patch_is_consumed_atomically() {
    let source = Harness::default();
    let patch = source
        .app
        .prepare_session_projection_patch(vec![hi_agent::SessionEvent::new(
            hi_agent::SessionEventKind::Message {
                message: hi_ai::Message::user("patched"),
            },
        )])
        .unwrap();
    let mut target = Harness::default();
    target.handle(Command::SessionPatch { patch }).unwrap();
    assert_eq!(target.app.session_projection_snapshot().revision, 1);
    assert!(target.app.transcript_text().contains("patched"));
}

#[test]
fn live_projection_assigns_stable_terminal_block_ids() {
    let mut harness = Harness::default();
    harness
        .handle(Command::Transcript {
            event: UiEvent::Text {
                text: "stable answer".into(),
            },
        })
        .unwrap();
    harness
        .handle(Command::Transcript {
            event: UiEvent::AssistantEnd,
        })
        .unwrap();
    let terminal_revision = harness.app.session_projection_snapshot().revision;
    harness
        .handle(Command::Transcript {
            event: UiEvent::AssistantEnd,
        })
        .unwrap();
    assert_eq!(
        harness.app.session_projection_snapshot().revision,
        terminal_revision,
        "replayed terminal UI events must be idempotent"
    );

    let first = harness.handle(Command::Inspect).unwrap();
    let second = harness.handle(Command::Inspect).unwrap();
    let first_block = &first["component_tree"]["root"]["children"][0]["children"][0];
    let second_block = &second["component_tree"]["root"]["children"][0]["children"][0];
    assert_eq!(first_block["id"], second_block["id"]);
    assert_eq!(first_block["attributes"]["stable_id"], "tui.assistant.1");
    assert_eq!(first_block["attributes"]["lifecycle"], "completed");
    assert_eq!(first["session_projection"]["revision"], 2);
}

#[test]
fn projection_gate_off_preserves_positional_transcript_behavior() {
    let mut app = harness_app("openai", "gpt-test");
    app.apply(UiEvent::Text {
        text: "legacy answer".into(),
    });
    app.apply(UiEvent::AssistantEnd);
    assert_eq!(app.session_projection_snapshot().revision, 0);
    assert!(
        app.projected_transcript_identities()
            .iter()
            .all(Option::is_none)
    );
    assert!(app.transcript_text().contains("legacy answer"));
}

#[test]
fn projection_patch_replay_is_idempotent_but_conflicting_terminal_is_rejected() {
    let id = hi_agent::TranscriptBlockId::new("turn.1.assistant").unwrap();
    let source = Harness::default();
    let patch = source
        .app
        .prepare_session_projection_patch(vec![
            hi_agent::SessionEvent::new(hi_agent::SessionEventKind::TranscriptBlockOpened {
                block_id: id.clone(),
                kind: hi_agent::TranscriptBlockKind::Assistant,
                content: "settled once".into(),
            }),
            hi_agent::SessionEvent::new(hi_agent::SessionEventKind::TranscriptBlockSettled {
                block_id: id.clone(),
                terminal: hi_agent::TranscriptBlockTerminal::Completed,
            }),
        ])
        .unwrap();
    let mut target = Harness::default();
    target
        .handle(Command::SessionPatch {
            patch: patch.clone(),
        })
        .unwrap();
    let before = target.app.session_projection_snapshot();
    let mut forged = patch.clone();
    forged.events.clear();
    let forged_error = target
        .handle(Command::SessionPatch { patch: forged })
        .unwrap_err();
    assert_eq!(forged_error.code, "invalid_session_projection");
    target.handle(Command::SessionPatch { patch }).unwrap();
    let after = target.app.session_projection_snapshot();
    assert_eq!(after.revision, before.revision);
    assert_eq!(after.digest, before.digest);

    let error = target
        .handle(Command::SessionEvent {
            event: hi_agent::SessionEvent::new(
                hi_agent::SessionEventKind::TranscriptBlockSettled {
                    block_id: id,
                    terminal: hi_agent::TranscriptBlockTerminal::Failed,
                },
            ),
        })
        .unwrap_err();
    assert_eq!(error.code, "invalid_session_projection");
    assert!(error.message.contains("already settled"));
    assert_eq!(
        target.app.session_projection_snapshot().digest,
        before.digest
    );
}

#[test]
fn transcript_block_snapshot_and_tail_patch_rebuild_the_same_view() {
    let id = hi_agent::TranscriptBlockId::new("snapshot.assistant").unwrap();
    let source = Harness::default();
    let open = source
        .app
        .prepare_session_projection_patch(vec![hi_agent::SessionEvent::new(
            hi_agent::SessionEventKind::TranscriptBlockOpened {
                block_id: id.clone(),
                kind: hi_agent::TranscriptBlockKind::Assistant,
                content: "snapshot".into(),
            },
        )])
        .unwrap();
    let mut source = source;
    source
        .handle(Command::SessionPatch { patch: open })
        .unwrap();
    let snapshot = source.app.session_projection_snapshot();

    let mut restored = Harness::default();
    restored
        .handle(Command::SessionSnapshot {
            snapshot: Box::new(snapshot),
        })
        .unwrap();
    let tail = restored
        .app
        .prepare_session_projection_patch(vec![hi_agent::SessionEvent::new(
            hi_agent::SessionEventKind::TranscriptBlockSettled {
                block_id: id,
                terminal: hi_agent::TranscriptBlockTerminal::Completed,
            },
        )])
        .unwrap();
    restored
        .handle(Command::SessionPatch { patch: tail })
        .unwrap();
    let inspected = restored.handle(Command::Inspect).unwrap();
    let block = &inspected["component_tree"]["root"]["children"][0]["children"][0];
    assert_eq!(block["attributes"]["stable_id"], "snapshot.assistant");
    assert_eq!(block["attributes"]["lifecycle"], "completed");
    assert_eq!(block["text"], "snapshot");
}

#[test]
fn protocol_errors_are_typed_and_do_not_end_the_stream() {
    let rows = drive(concat!(
        "not-json\n",
        "{\"id\":2,\"command\":\"resize\",\"width\":0,\"height\":20}\n",
        "{\"id\":3,\"command\":\"key\",\"key\":\"not-a-key\"}\n",
        "{\"id\":4,\"command\":\"hello\"}\n",
        "{\"id\":5,\"command\":\"resize\",\"width\":1,\"height\":1}\n",
        "{\"id\":6,\"command\":\"render\"}\n",
    ));
    assert_eq!(rows[0]["error"]["code"], "invalid_json");
    assert_eq!(rows[1]["error"]["code"], "invalid_dimensions");
    assert_eq!(rows[2]["error"]["code"], "invalid_key");
    assert_eq!(rows[3]["ok"], true);
    assert_eq!(rows[5]["result"]["lines"].as_array().unwrap().len(), 1);
}
