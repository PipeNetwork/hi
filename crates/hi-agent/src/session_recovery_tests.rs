use super::*;

fn event(kind: SessionEventKind) -> SessionEvent {
    SessionEvent::new(kind)
}

fn message(text: &str) -> SessionEvent {
    event(SessionEventKind::Message {
        message: Message::user(text),
    })
}

fn execution(id: &str) -> crate::WorkspaceTranscriptExecution {
    crate::WorkspaceTranscriptExecution {
        schema_version: crate::WorkspaceTranscriptExecution::SCHEMA_VERSION,
        operation_id: hi_workspace::OperationId::new(id),
        assistant_content: vec![hi_ai::Content::ToolCall {
            id: id.into(),
            name: "write".into(),
            arguments: "{}".into(),
        }],
        calls: vec![crate::WorkspaceTranscriptCall {
            call_id: id.into(),
            name: "write".into(),
            result: "written".into(),
        }],
        execution: hi_workspace::ExecutionReport::succeeded(None),
    }
}

fn staged(execution: &crate::WorkspaceTranscriptExecution) -> SessionEvent {
    event(SessionEventKind::WorkspaceExecutionStaged {
        execution: execution.clone(),
        visible_on_resume: true,
    })
}

fn settled(execution: &crate::WorkspaceTranscriptExecution) -> SessionEvent {
    event(SessionEventKind::WorkspaceExecutionSettled {
        operation_id: execution.operation_id.clone(),
    })
}

fn restore(reducer: &SessionReducer) -> SessionReducer {
    let bytes = serde_json::to_vec(&reducer.snapshot()).unwrap();
    SessionReducer::from_snapshot(serde_json::from_slice(&bytes).unwrap()).unwrap()
}

#[test]
fn settlement_credit_is_one_validated_replay_transition() {
    let mut reducer = SessionReducer::new();
    let mut before = crate::TaskRecoveryState::new("finish".into(), 3);
    before.remaining = 1;
    reducer
        .apply_event(event(SessionEventKind::TaskRecovery {
            state: before.clone(),
        }))
        .unwrap();
    let mut credited = before.clone();
    credited.observe_required_effect("goal:finish:0".into());
    assert!(
        SessionEvent::decode_remote_record(
            "turn_outcome",
            &serde_json::json!({"type":"task_recovery", "state":credited}).to_string(),
        )
        .is_err(),
        "a recovery-only payload cannot masquerade as its terminal receipt"
    );
    let mut goal = Goal::new("finish", vec!["implement".into()]);
    reducer
        .apply_event(event(SessionEventKind::Goal { goal: goal.clone() }))
        .unwrap();
    goal.advance();
    let snapshot = reducer.snapshot();
    let mut future = credited.clone();
    future.schema_version += 1;
    assert!(
        reducer
            .apply_event(event(SessionEventKind::TurnOutcome {
                status: TurnStatus::Completed,
                stop_reason: TurnStopReason::Completed,
                task_recovery: Some(future),
                settled_goal: Some(Box::new(goal.clone())),
            }))
            .is_err()
    );
    assert_eq!(
        serde_json::to_value(reducer.snapshot()).unwrap(),
        serde_json::to_value(&snapshot).unwrap()
    );
    assert_eq!(reducer.through_sequence(), snapshot.through_sequence);
    // An old outcome carries no credit and remains readable.
    reducer
        .apply_event(
            SessionEvent::decode_legacy_json(
                r#"{"type":"turn_outcome","status":"completed","stop_reason":"completed"}"#,
            )
            .unwrap()
            .unwrap(),
        )
        .unwrap();
    assert_eq!(reducer.state().task_recovery, before);
    reducer
        .apply_event(event(SessionEventKind::TurnOutcome {
            status: TurnStatus::Completed,
            stop_reason: TurnStopReason::Completed,
            task_recovery: Some(credited.clone()),
            settled_goal: Some(Box::new(goal.clone())),
        }))
        .unwrap();
    assert_eq!(restore(&reducer).state().task_recovery, credited);
    assert_eq!(restore(&reducer).state().goal.as_ref(), Some(&goal));
}

#[test]
fn workspace_recovery_snapshot_and_tail_match_every_crash_boundary() {
    let execution = execution("write-once");
    let events = [
        message("before"),
        staged(&execution),
        settled(&execution),
        event(SessionEventKind::Message {
            message: Message::assistant(execution.assistant_content.clone()),
        }),
        event(SessionEventKind::Message {
            message: Message::tool_result("write-once", "written"),
        }),
        message("after"),
    ];
    for crash in 0..=events.len() {
        let mut full = SessionReducer::new();
        full.apply_all(events[..crash].iter().cloned()).unwrap();
        let (expected, recovered) = full.into_restored_state();
        if crash == 2 {
            assert!(
                expected.messages[1]
                    .text()
                    .contains("workspace recovery pending")
            );
            assert!(!recovered);
        } else if crash >= 3 {
            assert_eq!(expected.messages.len(), if crash == 6 { 4 } else { 3 });
            assert_eq!(recovered, crash < 5);
        }
        for snapshot_at in 0..=crash {
            let mut prefix = SessionReducer::new();
            prefix
                .apply_all(events[..snapshot_at].iter().cloned())
                .unwrap();
            for _ in 0..2 {
                let mut restored = restore(&prefix);
                restored
                    .apply_all(events[snapshot_at..crash].iter().cloned())
                    .unwrap();
                let (actual, actual_recovered) = restored.into_restored_state();
                assert!(
                    actual.semantically_eq(&expected),
                    "snapshot {snapshot_at}, crash {crash}"
                );
                assert_eq!(actual_recovered, recovered);
            }
        }
    }
}

#[test]
fn recovery_survives_drive_compaction_rewind_and_snapshot_until_explicitly_replaced() {
    let mut recovery = crate::TaskRecoveryState::new("same objective".into(), 3);
    assert!(recovery.intervene("review"));
    assert!(recovery.intervene("protocol"));
    let mut reducer = SessionReducer::new();
    reducer
        .apply_event(event(SessionEventKind::TaskRecovery {
            state: recovery.clone(),
        }))
        .unwrap();
    reducer
        .apply_event(message(crate::PLAN_DRIVE_PROMPT))
        .unwrap();
    reducer
        .apply_event(event(SessionEventKind::Compaction {
            messages: vec![Message::user("summary")],
        }))
        .unwrap();
    reducer
        .apply_event(event(SessionEventKind::StateReplacement {
            messages: vec![],
            goal: None,
            decisions: vec![],
            plan: vec![],
        }))
        .unwrap();
    let mut restored = restore(&reducer);
    assert_eq!(restored.state().task_recovery, recovery);
    restored
        .apply_event(event(SessionEventKind::TaskRecovery {
            state: crate::TaskRecoveryState::default(),
        }))
        .unwrap();
    assert_eq!(
        restored.state().task_recovery,
        crate::TaskRecoveryState::default()
    );
}

#[test]
fn compaction_retires_settled_effects_but_keeps_pending_boundary_before_later_input() {
    let mut reducer = SessionReducer::new();
    for text in ["old one", "old two", "old three"] {
        reducer.apply_event(message(text)).unwrap();
    }
    let acknowledged = execution("acknowledged");
    reducer.apply_event(staged(&acknowledged)).unwrap();
    reducer.apply_event(settled(&acknowledged)).unwrap();
    reducer.apply_event(staged(&execution("pending"))).unwrap();
    reducer
        .apply_event(event(SessionEventKind::Compaction {
            messages: vec![Message::user("summary")],
        }))
        .unwrap();
    reducer.apply_event(message("later user input")).unwrap();
    let (state, recovered) = restore(&reducer).into_restored_state();
    assert!(!recovered);
    assert_eq!(state.messages.len(), 3);
    assert_eq!(state.messages[0].text(), "summary");
    assert!(state.messages[1].text().contains("Operation pending"));
    assert_eq!(state.messages[2].text(), "later user input");
}

#[test]
fn duplicate_stages_are_idempotent_and_conflicting_evidence_is_rejected_atomically() {
    let mut reducer = SessionReducer::new();
    let first = execution("same-operation");
    reducer.apply_event(staged(&first)).unwrap();
    reducer.apply_event(staged(&first)).unwrap();
    let before = serde_json::to_value(reducer.snapshot()).unwrap();
    let mut conflicting = first.clone();
    conflicting.calls[0].result = "different".into();
    assert!(reducer.apply_event(staged(&conflicting)).is_err());
    assert_eq!(serde_json::to_value(reducer.snapshot()).unwrap(), before);
    reducer.apply_event(settled(&first)).unwrap();
    reducer.apply_event(settled(&first)).unwrap();
    let second = execution("second-operation");
    reducer.apply_event(staged(&second)).unwrap();
    reducer.apply_event(settled(&second)).unwrap();
    let (state, _) = reducer.into_restored_state();
    assert_eq!(
        state.messages.len(),
        4,
        "each distinct operation materializes exactly once at the shared boundary"
    );
}

#[test]
fn recognized_invalid_recovery_and_future_envelopes_cannot_reset_the_budget() {
    let mut reducer = SessionReducer::new();
    let mut recovery = crate::TaskRecoveryState::default();
    recovery.intervene("fix");
    reducer
        .apply_event(event(SessionEventKind::TaskRecovery {
            state: recovery.clone(),
        }))
        .unwrap();
    let mut future = recovery.clone();
    future.schema_version += 1;
    let mut inconsistent = recovery.clone();
    inconsistent.remaining = inconsistent.limit + 1;
    for invalid in [future, inconsistent] {
        assert!(
            reducer
                .apply_event(event(SessionEventKind::TaskRecovery {
                    state: invalid.clone()
                }))
                .is_err()
        );
        let payload = serde_json::json!({"type":"task_recovery", "state": invalid}).to_string();
        let decoded = SessionEvent::decode_legacy_json(&payload).unwrap().unwrap();
        assert!(reducer.apply_event(decoded).is_err());
        assert_eq!(reducer.state().task_recovery, recovery);
        assert_eq!(reducer.through_sequence(), 1);
    }
    for payload in [
        r#"{"type":"task_recovery","state":{}}"#,
        r#"{"schema_version":2,"kind":{"type":"opaque_boundary"}}"#,
    ] {
        assert!(SessionEvent::decode_legacy_json(payload).is_err());
        assert!(SessionEvent::decode_remote_record("task_recovery", payload).is_err());
    }
    assert!(
        SessionEvent::decode_remote_record("task_recovery", r#"{"type":"goal_cleared"}"#).is_err()
    );
    for id in [".", "..", "../escape"] {
        assert!(
            reducer
                .apply_event(event(SessionEventKind::RemoteSessionIdentity {
                    session_id: id.into()
                }))
                .is_err()
        );
    }
    let mut snapshot = reducer.snapshot();
    snapshot.state.task_recovery.remaining = snapshot.state.task_recovery.limit + 1;
    assert!(SessionReducer::from_snapshot(snapshot).is_err());
}

#[test]
fn large_transcript_replay_restores_indexes_and_preserves_targeted_lifecycle() {
    let mut reducer = SessionReducer::new();
    for index in 0..20_000 {
        let id = TranscriptBlockId::new(format!("block:{index}")).unwrap();
        reducer
            .apply_event(event(SessionEventKind::TranscriptBlockRecorded {
                block_id: id,
                kind: TranscriptBlockKind::Assistant,
                content: "old".into(),
                terminal: TranscriptBlockTerminal::Completed,
            }))
            .unwrap();
    }
    let id = TranscriptBlockId::new("current").unwrap();
    reducer
        .apply_event(event(SessionEventKind::TranscriptBlockOpened {
            block_id: id.clone(),
            kind: TranscriptBlockKind::Assistant,
            content: String::new(),
        }))
        .unwrap();
    let mut restored = restore(&reducer);
    for _ in 0..2_000 {
        restored
            .apply_event(event(SessionEventKind::TranscriptBlockAppended {
                block_id: id.clone(),
                delta: "x".into(),
            }))
            .unwrap();
    }
    restored
        .apply_event(event(SessionEventKind::TranscriptBlockSettled {
            block_id: id,
            terminal: TranscriptBlockTerminal::Completed,
        }))
        .unwrap();
    restored
        .apply_event(event(SessionEventKind::TurnOutcome {
            task_recovery: None,
            settled_goal: None,
            status: TurnStatus::Completed,
            stop_reason: TurnStopReason::Completed,
        }))
        .unwrap();
    assert_eq!(restored.state().transcript_blocks.len(), 20_001);
    assert_eq!(
        restored
            .state()
            .transcript_blocks
            .last()
            .unwrap()
            .content
            .len(),
        2_000
    );
    assert!(
        restored.state().transcript_blocks[..20_000]
            .iter()
            .all(|block| block.content == "old")
    );
}

/// Run manually with `session_replay_scaling --ignored --nocapture`.
#[test]
#[ignore = "manual 1x/2x/4x replay timing gate"]
fn session_replay_scaling() {
    for size in [2_000, 4_000, 8_000] {
        let records: Vec<String> = (0..size)
            .flat_map(|index| {
                let id = TranscriptBlockId::new(format!("block:{index}")).unwrap();
                [
                    event(SessionEventKind::Message {
                        message: Message::user("x".repeat(256)),
                    }),
                    event(SessionEventKind::TranscriptBlockOpened {
                        block_id: id.clone(),
                        kind: TranscriptBlockKind::Assistant,
                        content: String::new(),
                    }),
                    event(SessionEventKind::TranscriptBlockAppended {
                        block_id: id.clone(),
                        delta: "y".repeat(256),
                    }),
                    event(SessionEventKind::TranscriptBlockSettled {
                        block_id: id,
                        terminal: TranscriptBlockTerminal::Completed,
                    }),
                ]
                .into_iter()
                .map(|event| serde_json::to_string(&event).unwrap())
            })
            .collect();
        let mut elapsed = Vec::new();
        for _ in 0..7 {
            let before_hashes = crate::session_projection::digest_calls();
            let started = std::time::Instant::now();
            let mut reducer = SessionReducer::new();
            for record in &records {
                let event = SessionEvent::decode_legacy_json(record).unwrap().unwrap();
                reducer.apply_event(event).unwrap();
            }
            let (state, interrupted) = reducer.into_restored_state();
            elapsed.push(started.elapsed());
            assert!(!interrupted);
            assert_eq!(state.messages.len(), size);
            assert_eq!(state.transcript_blocks.len(), size);
            assert_eq!(
                crate::session_projection::digest_calls() - before_hashes,
                0,
                "replay must never hash the full state per event"
            );
            std::hint::black_box(state);
        }
        elapsed.sort_unstable();
        eprintln!(
            "session replay: histories={size}, events={}, median_ms={:.3}, full_state_hashes=0",
            records.len(),
            elapsed[3].as_secs_f64() * 1000.0
        );
    }
}

#[test]
fn pending_validation_correction_survives_snapshot_and_compaction() {
    let mut recovery = crate::TaskRecoveryState::new("repair the failing test".into(), 3);
    recovery.observe(&crate::recovery::ValidationObservation::command(
        "test-execution".into(),
        "cargo test --workspace",
        "revision".into(),
        crate::recovery::ValidationResult::Failed,
        "test suite::failure ... FAILED",
        std::path::Path::new("."),
        false,
    ));
    assert_eq!(
        serde_json::to_value(&recovery).unwrap()["pending_validation_correction"],
        true
    );
    let mut reducer = SessionReducer::new();
    reducer
        .apply_event(event(SessionEventKind::TaskRecovery {
            state: recovery.clone(),
        }))
        .unwrap();
    reducer
        .apply_event(event(SessionEventKind::Compaction {
            messages: vec![Message::user("retained objective")],
        }))
        .unwrap();
    let restored = restore(&reducer);
    assert_eq!(
        serde_json::to_value(&restored.state().task_recovery).unwrap(),
        serde_json::to_value(&recovery).unwrap(),
    );
}
