use super::*;
use crate::TranscriptBlockLifecycle;

fn event(kind: SessionEventKind) -> SessionEvent {
    SessionEvent::new(kind)
}

fn message_texts(state: &SessionState) -> Vec<String> {
    state.messages.iter().map(Message::text).collect()
}

fn block_id(value: &str) -> TranscriptBlockId {
    TranscriptBlockId::new(value).unwrap()
}

#[test]
fn legacy_json_and_remote_records_reduce_to_the_same_projection() {
    let records = [
        serde_json::to_string(&Message::user("ship it")).unwrap(),
        r#"{"type":"usage","input_tokens":17,"output_tokens":4,"cache_read_tokens":3,"estimated":true}"#.to_owned(),
        r#"{"type":"name","name":"  Harness work  "}"#.to_owned(),
        r#"{"type":"pipe_fs_mode","enabled":true}"#.to_owned(),
    ];

    let mut local = SessionReducer::new();
    for line in &records {
        local
            .apply(SessionEvent::from_legacy_json(line).unwrap())
            .unwrap();
    }

    let mut remote = SessionReducer::new();
    remote
        .apply(SessionEvent::from_remote_record("message", &records[0]))
        .unwrap();
    for line in &records[1..] {
        remote
            .apply(SessionEvent::from_remote_record("metadata", line))
            .unwrap();
    }

    assert!(local.state().semantically_eq(remote.state()));
    assert_eq!(message_texts(local.state()), ["ship it"]);
    assert_eq!(local.state().usage.input_tokens, 17);
    assert_eq!(local.state().usage.context_occupancy, 17);
    assert_eq!(local.state().name.as_deref(), Some("Harness work"));
    assert_eq!(local.state().pipefs_enabled, Some(true));
}

#[test]
fn reduction_applies_last_write_and_delta_rules() {
    let valid_a = "a".repeat(64);
    let valid_b = "b".repeat(64);
    let duplicate = Decision {
        summary: "transport".into(),
        rationale: "old".into(),
        files: vec![],
    };
    let replacement = Decision {
        summary: "transport".into(),
        rationale: "new".into(),
        files: vec!["src/lib.rs".into()],
    };
    let mut reducer = SessionReducer::new();
    reducer
        .apply(event(SessionEventKind::Decisions {
            decisions: vec![duplicate, replacement],
        }))
        .unwrap();
    reducer
        .apply(event(SessionEventKind::PlanDrive {
            paused: true,
            resume_on_user_input: Some(false),
            stall: 3,
            evidence_reset: false,
            evidence_add: vec![valid_b.clone(), "not-a-digest".into(), valid_a.clone()],
        }))
        .unwrap();
    reducer
        .apply(event(SessionEventKind::PlanDrive {
            paused: false,
            resume_on_user_input: Some(false),
            stall: 0,
            evidence_reset: true,
            evidence_add: vec![valid_b.clone()],
        }))
        .unwrap();

    let state = reducer.state();
    assert_eq!(state.decisions.len(), 1);
    assert_eq!(state.decisions[0].rationale, "new");
    assert_eq!(
        state
            .plan_drive_evidence
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        [valid_b]
    );
    assert!(!state.plan_drive_paused);
}

#[test]
fn legacy_interruption_pause_is_inferred_then_consumed_by_real_user_turn() {
    let base = Message::user("original task");
    let synthetic = Message::user(crate::PLAN_DRIVE_PROMPT);
    let mut reducer = SessionReducer::new();
    reducer
        .apply(event(SessionEventKind::Message {
            message: base.clone(),
        }))
        .unwrap();
    reducer
        .apply(event(SessionEventKind::Message { message: synthetic }))
        .unwrap();
    reducer
        .apply(event(SessionEventKind::StateReplacement {
            messages: vec![base],
            goal: None,
            decisions: vec![],
            plan: vec![],
        }))
        .unwrap();
    reducer
        .apply(event(SessionEventKind::TurnOutcome {
            status: TurnStatus::Cancelled,
            stop_reason: TurnStopReason::Cancelled,
        }))
        .unwrap();
    let inferred = reducer
        .apply(event(SessionEventKind::PlanDrive {
            paused: true,
            resume_on_user_input: None,
            stall: 2,
            evidence_reset: false,
            evidence_add: vec!["c".repeat(64)],
        }))
        .unwrap();
    assert!(inferred.plan_drive_resume_on_user_input);

    reducer
        .apply(event(SessionEventKind::Message {
            message: Message::user("continue now"),
        }))
        .unwrap();
    let consumed = reducer
        .apply(event(SessionEventKind::TurnOutcome {
            status: TurnStatus::Completed,
            stop_reason: TurnStopReason::Completed,
        }))
        .unwrap();
    assert!(!consumed.plan_drive_paused);
    assert!(!consumed.plan_drive_resume_on_user_input);
    assert_eq!(consumed.plan_drive_stall, 0);
    assert!(consumed.plan_drive_evidence.is_empty());
}

#[test]
fn opaque_record_breaks_legacy_pause_inference() {
    let base = Message::user("base");
    let mut reducer = SessionReducer::new();
    reducer
        .apply(event(SessionEventKind::Message {
            message: base.clone(),
        }))
        .unwrap();
    reducer
        .apply(event(SessionEventKind::Message {
            message: Message::user(crate::PLAN_DRIVE_PROMPT),
        }))
        .unwrap();
    reducer
        .apply(event(SessionEventKind::StateReplacement {
            messages: vec![base],
            goal: None,
            decisions: vec![],
            plan: vec![],
        }))
        .unwrap();
    reducer.apply(SessionEvent::opaque_boundary()).unwrap();
    let state = reducer
        .apply(event(SessionEventKind::PlanDrive {
            paused: true,
            resume_on_user_input: None,
            stall: 1,
            evidence_reset: false,
            evidence_add: vec![],
        }))
        .unwrap();
    assert!(!state.plan_drive_resume_on_user_input);
}

#[test]
fn sequence_and_version_errors_leave_state_unchanged() {
    let mut reducer = SessionReducer::new();
    reducer
        .apply(
            event(SessionEventKind::Message {
                message: Message::user("one"),
            })
            .at_sequence(1),
        )
        .unwrap();

    let error = reducer
        .apply(
            event(SessionEventKind::Message {
                message: Message::user("gap"),
            })
            .at_sequence(3),
        )
        .unwrap_err();
    assert_eq!(
        error,
        SessionReduceError::NonContiguousSequence {
            expected: 2,
            found: 3,
        }
    );
    assert_eq!(message_texts(reducer.state()), ["one"]);
    assert_eq!(reducer.through_sequence(), 1);

    let mut future = event(SessionEventKind::OpaqueBoundary);
    future.schema_version += 1;
    assert!(matches!(
        reducer.apply(future),
        Err(SessionReduceError::UnsupportedEventVersion { .. })
    ));
    assert_eq!(reducer.through_sequence(), 1);
}

#[test]
fn serialized_snapshot_plus_tail_matches_full_replay() {
    let events = vec![
        event(SessionEventKind::Message {
            message: Message::user("first"),
        })
        .at_sequence(1),
        event(SessionEventKind::Name {
            name: "work".into(),
        })
        .at_sequence(2),
        event(SessionEventKind::Message {
            message: Message::assistant(vec![]),
        })
        .at_sequence(3),
        event(SessionEventKind::Checkpoints {
            refs: vec!["refs/hi/checkpoints/1".into()],
        })
        .at_sequence(4),
    ];

    let mut full = SessionReducer::new();
    full.apply_all(events.clone()).unwrap();

    let mut partial = SessionReducer::new();
    partial.apply_all(events[..2].iter().cloned()).unwrap();
    let encoded = serde_json::to_vec(&partial.snapshot()).unwrap();
    let snapshot: SessionReducerSnapshot = serde_json::from_slice(&encoded).unwrap();
    let mut restored = SessionReducer::from_snapshot(snapshot).unwrap();
    restored.apply_all(events[2..].iter().cloned()).unwrap();

    assert_eq!(restored.through_sequence(), 4);
    assert!(restored.state().semantically_eq(full.state()));
}

#[test]
fn legacy_decoder_preserves_unknown_records_as_boundaries() {
    assert!(SessionEvent::from_legacy_json(" \n ").is_none());
    assert!(matches!(
        SessionEvent::from_legacy_json(r#"{"type":"future_record","x":1}"#)
            .unwrap()
            .kind,
        SessionEventKind::OpaqueBoundary
    ));
    assert!(matches!(
        SessionEvent::from_remote_record(
            "metadata",
            &serde_json::to_string(&Message::user("not a metadata record")).unwrap(),
        )
        .kind,
        SessionEventKind::OpaqueBoundary
    ));
}

#[test]
fn invalid_identity_and_future_snapshot_are_rejected() {
    let mut reducer = SessionReducer::new();
    assert_eq!(
        reducer
            .apply(event(SessionEventKind::RemoteSessionIdentity {
                session_id: "../escape".into(),
            }))
            .unwrap_err(),
        SessionReduceError::InvalidRemoteSessionIdentity
    );
    assert_eq!(reducer.through_sequence(), 0);

    let mut snapshot = reducer.snapshot();
    snapshot.reducer_version += 1;
    assert!(matches!(
        SessionReducer::from_snapshot(snapshot),
        Err(SessionReduceError::UnsupportedReducerVersion { .. })
    ));
}

#[test]
fn transcript_blocks_have_stable_identity_and_exactly_once_settlement() {
    let id = block_id("turn-7:assistant-1");
    let mut reducer = SessionReducer::new();
    reducer
        .apply(event(SessionEventKind::TranscriptBlockOpened {
            block_id: id.clone(),
            kind: TranscriptBlockKind::Assistant,
            content: "hel".into(),
        }))
        .unwrap();
    reducer
        .apply(event(SessionEventKind::TranscriptBlockAppended {
            block_id: id.clone(),
            delta: "lo".into(),
        }))
        .unwrap();
    reducer
        .apply(event(SessionEventKind::TranscriptBlockSettled {
            block_id: id.clone(),
            terminal: TranscriptBlockTerminal::Completed,
        }))
        .unwrap();

    let block = &reducer.state().transcript_blocks[0];
    assert_eq!(block.id, id);
    assert_eq!(block.content, "hello");
    assert_eq!(block.opened_sequence, 1);
    assert_eq!(block.last_transition_sequence, 3);
    assert_eq!(
        block.lifecycle,
        TranscriptBlockLifecycle::Settled {
            terminal: TranscriptBlockTerminal::Completed,
            sequence: 3,
        }
    );

    let before = reducer.snapshot();
    assert!(matches!(
        reducer.apply(event(SessionEventKind::TranscriptBlockSettled {
            block_id: block_id("turn-7:assistant-1"),
            terminal: TranscriptBlockTerminal::Failed,
        })),
        Err(SessionReduceError::TranscriptBlock(
            TranscriptBlockTransitionError::AlreadySettled { .. }
        ))
    ));
    assert_eq!(reducer.through_sequence(), before.through_sequence);
    assert_eq!(
        reducer.snapshot().state.transcript_blocks,
        before.state.transcript_blocks
    );
}

#[test]
fn turn_outcome_requires_all_open_transcript_blocks_to_settle() {
    let id = block_id("tool-call-42");
    let mut reducer = SessionReducer::new();
    reducer
        .apply(event(SessionEventKind::TranscriptBlockOpened {
            block_id: id.clone(),
            kind: TranscriptBlockKind::Tool,
            content: "running".into(),
        }))
        .unwrap();
    let before = reducer.snapshot();

    assert!(matches!(
        reducer.apply(event(SessionEventKind::TurnOutcome {
            status: TurnStatus::Completed,
            stop_reason: TurnStopReason::Completed,
        })),
        Err(SessionReduceError::TranscriptBlock(
            TranscriptBlockTransitionError::UnsettledAtTurnEnd { .. }
        ))
    ));
    assert_eq!(reducer.through_sequence(), before.through_sequence);

    reducer
        .apply(event(SessionEventKind::TranscriptBlockSettled {
            block_id: id,
            terminal: TranscriptBlockTerminal::Cancelled,
        }))
        .unwrap();
    reducer
        .apply(event(SessionEventKind::TurnOutcome {
            status: TurnStatus::Cancelled,
            stop_reason: TurnStopReason::Cancelled,
        }))
        .unwrap();
}

#[test]
fn legacy_block_records_decode_aliases_and_old_snapshots_default_empty() {
    let opened = SessionEvent::from_legacy_json(
        r#"{"type":"transcript_block_opened","id":"legacy:block-1","kind":"reasoning","text":"why"}"#,
    )
    .unwrap();
    let settled = SessionEvent::from_legacy_json(
        r#"{"type":"transcript_block_settled","id":"legacy:block-1","terminal":"completed"}"#,
    )
    .unwrap();
    let mut reducer = SessionReducer::new();
    reducer.apply(opened).unwrap();
    reducer.apply(settled).unwrap();
    assert_eq!(reducer.state().transcript_blocks[0].content, "why");

    let legacy = SessionReducer::new().snapshot();
    let value = serde_json::to_value(legacy).unwrap();
    assert!(value["state"].get("transcript_blocks").is_none());
    let decoded: SessionReducerSnapshot = serde_json::from_value(value).unwrap();
    let restored = SessionReducer::from_snapshot(decoded).unwrap();
    assert!(restored.state().transcript_blocks.is_empty());
}

#[test]
fn snapshot_restore_rejects_duplicate_or_impossible_block_lifecycle() {
    let id = block_id("snapshot:block");
    let mut reducer = SessionReducer::new();
    reducer
        .apply(event(SessionEventKind::TranscriptBlockRecorded {
            block_id: id,
            kind: TranscriptBlockKind::Notice,
            content: "done".into(),
            terminal: TranscriptBlockTerminal::Completed,
        }))
        .unwrap();
    let mut snapshot = reducer.snapshot();
    snapshot.state.transcript_blocks[0].last_transition_sequence = 0;
    assert!(matches!(
        SessionReducer::from_snapshot(snapshot),
        Err(SessionReduceError::TranscriptBlock(
            TranscriptBlockTransitionError::InvalidSnapshot { .. }
        ))
    ));
}
