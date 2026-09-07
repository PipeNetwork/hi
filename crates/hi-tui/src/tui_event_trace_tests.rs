use std::io::Read;

use super::*;

fn wire_audit_with_secret() -> hi_ai::WireAudit {
    hi_ai::WireAudit {
        provider: "openai_compatible".into(),
        route: "chat_completions".into(),
        model: "pipe/test".into(),
        output_token_parameter: "max_tokens".into(),
        max_output_tokens: 512,
        temperature: Some(0.2),
        reasoning_request: Some("high".into()),
        native_tools_enabled: true,
        tool_count: 7,
        strict_schema: true,
        tool_choice: Some("auto".into()),
        request_attempt: 2,
        compatibility_fallback: Some("stream_usage".into()),
        accepted: true,
        request_body: Some(json!({
            "messages": [{"role": "user", "content": "private prompt"}],
            "authorization": "Bearer private-key"
        })),
        response_status: Some(200),
        ..hi_ai::WireAudit::default()
    }
}

#[test]
fn trace_is_versioned_sequenced_and_flushed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested/events.jsonl");
    let trace = TuiEventTrace::open_with_run_id(&path, None).unwrap();
    trace.emit("ready", json!({"width": 80})).unwrap();
    trace
        .emit_ui_event(&UiEvent::Text {
            text: "secret body".into(),
        })
        .unwrap();

    let mut contents = String::new();
    File::open(path)
        .unwrap()
        .read_to_string(&mut contents)
        .unwrap();
    let rows = contents
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["schema_version"], 1);
    assert_eq!(rows[0]["sequence"], 0);
    assert!(rows[0].get("run_id").is_none());
    assert_eq!(rows[1]["sequence"], 1);
    assert_eq!(rows[1]["event"], "ui_event");
    assert_eq!(rows[1]["data"]["kind"], "text");
    assert_eq!(rows[1]["data"]["chars"], 11);
    assert!(!contents.contains("secret body"));
}

#[test]
fn reopening_trace_appends_and_continues_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");
    {
        let trace = TuiEventTrace::open_with_run_id(&path, Some("123-18fdb42-1".into())).unwrap();
        trace.emit("ready", json!({})).unwrap();
        trace.emit("session_ended", json!({})).unwrap();
    }
    {
        let trace = TuiEventTrace::open_with_run_id(&path, Some("123-18fdb42-2".into())).unwrap();
        trace.emit("ready", json!({})).unwrap();
    }
    let rows = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["sequence"], 0);
    assert_eq!(rows[1]["sequence"], 1);
    assert_eq!(rows[2]["sequence"], 2);
    assert!(rows.iter().all(|row| row["process_id"].is_u64()));
    assert_eq!(rows[0]["run_id"], "123-18fdb42-1");
    assert_eq!(rows[1]["run_id"], "123-18fdb42-1");
    assert_eq!(rows[2]["run_id"], "123-18fdb42-2");
}

#[test]
fn smoke_run_marker_validation_accepts_only_harness_shape() {
    assert!(valid_smoke_run_marker("123-18fdb42-7"));
    for marker in [
        "",
        "0-18fdb42-7",
        "123-0-7",
        "123-18fdb42-0",
        "123-not-hex-7",
        "123-18fdb42",
        "123-18fdb42-7-extra",
    ] {
        assert!(!valid_smoke_run_marker(marker), "accepted {marker:?}");
    }
}

#[test]
fn reopening_after_partial_record_separates_new_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");
    std::fs::write(
        &path,
        b"{\"schema_version\":1,\"sequence\":7,\"event\":\"truncated",
    )
    .unwrap();
    let trace = TuiEventTrace::open(&path).unwrap();
    trace.emit("ready", json!({})).unwrap();
    let contents = std::fs::read_to_string(path).unwrap();
    let last = contents.lines().last().unwrap();
    let row: Value = serde_json::from_str(last).unwrap();
    assert_eq!(row["event"], "ready");
    assert_eq!(row["sequence"], 0);
}

#[test]
fn open_and_deferred_write_failures_are_explicit() {
    let dir = tempfile::tempdir().unwrap();
    let open_error = TuiEventTrace::open(dir.path()).err().unwrap();
    assert!(format!("{open_error:#}").contains("TUI event trace"));

    let trace = TuiEventTrace::open(dir.path().join("events.jsonl")).unwrap();
    trace.inner.lock().unwrap().failure = Some("simulated disk failure".into());
    assert!(format!("{:#}", trace.check().unwrap_err()).contains("simulated disk failure"));
    assert!(
        format!("{:#}", trace.emit("ready", json!({})).unwrap_err())
            .contains("simulated disk failure")
    );
}

#[test]
fn prompt_origins_cover_drive_user_and_command_paths() {
    assert_eq!(
        PromptOrigin::from_prompt(hi_agent::PLAN_DRIVE_PROMPT),
        PromptOrigin::PlanDrive
    );
    assert_eq!(
        PromptOrigin::from_prompt(hi_agent::GOAL_CONTINUE_PROMPT),
        PromptOrigin::GoalDrive
    );
    assert_eq!(
        PromptOrigin::from_prompt("/status"),
        PromptOrigin::CommandFollowUp
    );
    assert_eq!(PromptOrigin::from_prompt("fix it"), PromptOrigin::User);
}

#[test]
fn prompt_summary_correlates_without_exposing_prompt_text() {
    let first = prompt_summary("private prompt", PromptOrigin::User, 73);
    let repeated = prompt_summary("private prompt", PromptOrigin::User, 72);
    let different = prompt_summary("different prompt", PromptOrigin::User, 71);

    assert_eq!(first["queue_depth"], 73);
    assert_eq!(first["prompt_fingerprint"], repeated["prompt_fingerprint"]);
    assert_ne!(first["prompt_fingerprint"], different["prompt_fingerprint"]);
    assert!(
        !serde_json::to_string(&first)
            .unwrap()
            .contains("private prompt")
    );
}

#[test]
fn step_limit_summary_names_unlimited_without_exposing_the_sentinel() {
    assert_eq!(step_limit_summary(u32::MAX), json!({"mode": "unlimited"}));
    assert_eq!(
        step_limit_summary(2),
        json!({"mode": "finite", "max_steps": 2})
    );
}

#[test]
fn composed_tap_preserves_existing_sink_and_redacts_ui_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");
    let trace = TuiEventTrace::open(&path).unwrap();
    let seen = Arc::new(Mutex::new(0usize));
    let sink_seen = seen.clone();
    let base: crate::RemoteEventTap = Arc::new(move |_| {
        *sink_seen.lock().unwrap() += 1;
    });
    let tap = compose_remote_event_tap(Some(base), Some(trace)).unwrap();
    tap(&UiEvent::Reasoning {
        text: "private reasoning".into(),
    });
    assert_eq!(*seen.lock().unwrap(), 1);
    let contents = std::fs::read_to_string(path).unwrap();
    assert!(contents.contains("reasoning"));
    assert!(!contents.contains("private reasoning"));
}

#[test]
fn provider_request_trace_keeps_only_typed_scalar_audit_fields() {
    let audit = wire_audit_with_secret();
    let mut value = serde_json::to_value(audit).unwrap();
    value["future_nested_field"] = json!({"secret": "future-secret"});

    let summary = provider_request_summary(&value);
    assert_eq!(summary["provider"], "openai_compatible");
    assert_eq!(summary["model"], "pipe/test");
    assert_eq!(summary["request_attempt"], 2);
    assert_eq!(summary["response_status"], 200);
    assert!(summary.get("request_body").is_none());
    assert!(summary.get("future_nested_field").is_none());
    assert!(summary.as_object().unwrap().values().all(|value| matches!(
        value,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    )));
    let encoded = serde_json::to_string(&summary).unwrap();
    assert!(!encoded.contains("private prompt"));
    assert!(!encoded.contains("private-key"));
    assert!(!encoded.contains("future-secret"));
}

#[test]
fn provider_request_is_flushed_before_turn_settlement_without_renderable_secrets() {
    use hi_agent::Ui as _;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");
    let trace = TuiEventTrace::open(&path).unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let (confirmations, _confirmation_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut ui = crate::event::ChannelUi {
        tx,
        confirmations,
        event_sink: None,
        approval_store: None,
    };

    ui.provider_request(&wire_audit_with_secret());
    let event = rx.try_recv().expect("provider audit event");
    let serialized = serde_json::to_string(&event).unwrap();
    assert!(serialized.contains("provider_request"));
    assert!(!serialized.contains("private prompt"));
    assert!(!serialized.contains("private-key"));

    // The full-screen event tap performs this call while the turn future
    // is still running. No turn-settlement callback is involved.
    trace.emit_ui_event(&event).unwrap();
    let contents = std::fs::read_to_string(path).unwrap();
    let row: Value = serde_json::from_str(contents.trim()).unwrap();
    assert_eq!(row["event"], "provider_request");
    assert_eq!(row["data"]["response_status"], 200);
    assert!(!contents.contains("private prompt"));
    assert!(!contents.contains("private-key"));
}

#[test]
fn malformed_provider_audit_emits_safe_scalar_evidence() {
    assert_eq!(
        provider_request_summary(&json!({"request_body": {"secret": "value"}})),
        json!({"audit_valid": false})
    );
}
