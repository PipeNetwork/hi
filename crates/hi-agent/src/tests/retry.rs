use super::common::*;
use super::*;
use std::sync::Arc;

type CompactionRecords = Arc<Mutex<Vec<Vec<Message>>>>;
type StateReplacementRecords = Arc<Mutex<Vec<(Vec<Message>, Option<Goal>, Vec<Decision>)>>>;

struct CompactionRecordingSession {
    records: CompactionRecords,
}

struct StateReplacementRecordingSession {
    records: StateReplacementRecords,
}

impl SessionSink for CompactionRecordingSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, messages: &[Message]) -> anyhow::Result<()> {
        self.records.lock().unwrap().push(messages.to_vec());
        Ok(())
    }
}

impl SessionSink for StateReplacementRecordingSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_state_replacement(
        &mut self,
        messages: &[Message],
        goal: Option<&Goal>,
        decisions: &DecisionLog,
        _plan: &[crate::PlanStep],
    ) -> anyhow::Result<()> {
        self.records.lock().unwrap().push((
            messages.to_vec(),
            goal.cloned(),
            decisions.entries().to_vec(),
        ));
        Ok(())
    }
}

struct FailingCompactionSession;

impl SessionSink for FailingCompactionSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("disk full"))
    }
}

#[test]
fn durable_truncate_records_compaction_boundary() {
    let records = Arc::new(Mutex::new(Vec::new()));
    let mut agent = agent(vec![], config());
    agent.messages_mut().push(Message::user("old attempt"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    agent.set_session(Box::new(CompactionRecordingSession {
        records: records.clone(),
    }));

    agent.truncate_messages_durable(1).unwrap();

    assert_eq!(agent.messages().len(), 1);
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].len(), 1);
    assert_eq!(records[0][0].role, Role::System);
}

#[test]
fn durable_truncate_keeps_live_history_when_persistence_fails() {
    let mut agent = agent(vec![], config());
    agent.messages_mut().push(Message::user("old attempt"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    agent.set_session(Box::new(FailingCompactionSession));

    let err = agent.truncate_messages_durable(1).unwrap_err();

    assert!(err.to_string().contains("disk full"));
    assert_eq!(agent.messages().len(), 3);
    assert_eq!(agent.messages()[1].text(), "old attempt");
}

#[test]
fn retry_rewind_restores_state_snapshot_and_rebuilt_system_prompt() {
    let records = Arc::new(Mutex::new(Vec::new()));
    let mut agent = agent(vec![], config());
    agent.set_goal(Some("keep this goal".into()));
    agent.decisions.record(Decision {
        summary: "kept decision".into(),
        rationale: "pre-turn state".into(),
        files: vec!["src/lib.rs".into()],
    });
    agent.refresh_system_message();

    let start = agent.messages().len();
    let snapshot = agent.state_snapshot();

    agent.messages_mut().push(Message::user("old attempt"));
    agent.decisions.record(Decision {
        summary: "discarded decision".into(),
        rationale: "recorded during abandoned attempt".into(),
        files: vec!["src/bad.rs".into()],
    });
    agent.set_goal(Some("discarded goal".into()));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    agent.set_session(Box::new(StateReplacementRecordingSession {
        records: records.clone(),
    }));

    agent.rewind_to_snapshot_durable(start, &snapshot).unwrap();

    assert_eq!(agent.messages().len(), 1);
    assert_eq!(agent.goal(), Some("keep this goal"));
    assert_eq!(agent.decisions().entries().len(), 1);
    assert_eq!(agent.decisions().entries()[0].summary, "kept decision");
    let system = agent.messages()[0].text();
    assert!(system.contains("keep this goal"), "system prompt: {system}");
    assert!(system.contains("kept decision"), "system prompt: {system}");
    assert!(
        !system.contains("discarded decision") && !system.contains("discarded goal"),
        "discarded state leaked into system prompt: {system}"
    );

    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].0.len(), 1);
    assert!(records[0].0[0].text().contains("kept decision"));
    assert!(!records[0].0[0].text().contains("discarded decision"));
    assert!(records[0].1.is_none());
    assert_eq!(records[0].2.len(), 1);
    assert_eq!(records[0].2[0].summary, "kept decision");
}

#[tokio::test]
async fn request_too_large_drops_prior_context_and_retries_latest_prompt() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::RequestTooLarge,
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 12, 3)),
        ],
        config(),
    );
    let huge_old_output = "old tool output ".repeat(20_000);
    agent.messages_mut().push(Message::user("previous task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "read-1".into(),
            name: "read".into(),
            arguments: r#"{"path":"LICENSE"}"#.into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("read-1", huge_old_output.clone()));

    let mut ui = RecordingUi::default();
    agent
        .run_turn("fix the current bug", &mut ui)
        .await
        .unwrap();

    let requests = requests.lock().unwrap();
    let contains = |messages: &[Message], needle: &str| {
        messages.iter().flat_map(|m| &m.content).any(|c| match c {
            Content::Text(t) => t.contains(needle),
            Content::Thinking { text, .. } => text.contains(needle),
            Content::ToolCall {
                name, arguments, ..
            } => name.contains(needle) || arguments.contains(needle),
            Content::ToolResult { output, .. } => output.contains(needle),
            _ => false,
        })
    };
    assert_eq!(requests.len(), 2);
    assert!(
        contains(&requests[0], &huge_old_output),
        "first request includes existing context"
    );
    assert!(
        !contains(&requests[1], &huge_old_output),
        "retry omits oversized prior context"
    );
    assert!(
        requests[1]
            .iter()
            .any(|m| m.text().contains("fix the current bug")),
        "latest user request is preserved"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("dropped prior conversation context")),
        "user sees recovery status: {:?}",
        ui.statuses
    );
    assert_eq!(agent.messages().last().unwrap().text(), "ok");
}

#[tokio::test]
async fn request_too_large_context_drop_records_durable_boundary() {
    let records = Arc::new(Mutex::new(Vec::new()));
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::RequestTooLarge,
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 12, 3)),
        ],
        config(),
    );
    agent.messages_mut().push(Message::user("previous task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "old answer with huge context".repeat(1000),
        )]));
    agent.set_session(Box::new(CompactionRecordingSession {
        records: records.clone(),
    }));

    agent
        .run_turn("fix the current bug", &mut RecordingUi::default())
        .await
        .unwrap();

    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].len(),
        1,
        "context-drop boundary should persist only the rebuilt system prompt"
    );
    assert_eq!(records[0][0].role, Role::System);
    assert!(
        !records[0][0].text().contains("previous task"),
        "discarded context must not survive the durable boundary"
    );
}

#[tokio::test]
async fn request_too_large_keeps_live_context_when_boundary_persistence_fails() {
    let (mut agent, requests) = scripted_agent(vec![ProviderStep::RequestTooLarge], config());
    agent.messages_mut().push(Message::user("previous task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    let start_len = agent.messages().len();
    agent.set_session(Box::new(FailingCompactionSession));
    let mut ui = RecordingUi::default();

    let err = agent.run_turn("fix it", &mut ui).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::RequestTooLarge)
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        1,
        "failed durable boundary should abort recovery instead of retrying from divergent state"
    );
    assert_eq!(agent.messages().len(), start_len);
    assert_eq!(agent.messages()[1].text(), "previous task");
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("couldn't persist dropped-context retry state")),
        "user sees persistence failure: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn request_too_large_latest_prompt_is_removed_after_failed_retry() {
    let (mut agent, _requests) = scripted_agent(vec![ProviderStep::RequestTooLarge], config());
    let start_len = agent.messages().len();
    let mut ui = RecordingUi::default();

    let err = agent
        .run_turn(&"single huge prompt ".repeat(20_000), &mut ui)
        .await
        .unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::RequestTooLarge)
    );
    assert_eq!(
        agent.messages().len(),
        start_len,
        "failed oversized prompt is not left in live history"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("shorten the prompt")),
        "user gets actionable status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn request_too_large_failed_retry_after_dropping_context_removes_latest_prompt() {
    let (mut agent, requests) = scripted_agent(
        vec![ProviderStep::RequestTooLarge, ProviderStep::RequestTooLarge],
        config(),
    );
    agent.messages_mut().push(Message::user("previous task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "old answer with lots of prior context".into(),
        )]));
    let start_len = agent.messages().len();
    let huge_prompt = "still too large ".repeat(20_000);
    let mut ui = RecordingUi::default();

    let err = agent.run_turn(&huge_prompt, &mut ui).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::RequestTooLarge)
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "first oversized request should retry once with prior context dropped"
    );
    assert_eq!(
        agent.messages().len(),
        1,
        "failed retry should remove the rewritten latest prompt instead of leaving it in history"
    );
    assert!(
        agent
            .messages()
            .iter()
            .all(|message| !message.text().contains(&huge_prompt[..64])),
        "oversized latest prompt should not remain in live history"
    );
    assert!(
        start_len > agent.messages().len(),
        "test must exercise the context-dropping retry path"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("shorten the prompt")),
        "user gets actionable status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn context_preflight_rejects_hopeless_oversized_prompt_without_provider_call() {
    let mut cfg = config();
    cfg.routing.context_window = Some(1);
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    let (mut agent, requests) = scripted_agent(vec![], cfg);
    let start_len = agent.messages().len();
    let mut ui = RecordingUi::default();

    let err = agent.run_turn("x", &mut ui).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::RequestTooLarge)
    );
    assert!(
        requests.lock().unwrap().is_empty(),
        "locally impossible requests should not be sent to the provider"
    );
    assert_eq!(
        agent.messages().len(),
        start_len,
        "failed oversized prompt is not left in live history"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("shorten the prompt")),
        "user gets actionable status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn context_preflight_reduces_output_budget_to_available_headroom() {
    let mut cfg = config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.routing.max_tokens = 8192;
    cfg.routing.requested_max_tokens = 8192;
    let (mut agent, _requests, max_tokens) = scripted_agent_recording_max_tokens(
        vec![ProviderStep::Completion(completion(
            vec![Content::Text("ok".into())],
            10,
            2,
        ))],
        cfg,
    );
    let prompt = "hello";
    let prompt_estimate =
        hi_ai::estimate_messages_tokens(agent.messages()) + hi_ai::estimate_text_tokens(prompt);
    let expected_headroom = 2048;
    agent.set_model(
        "m".into(),
        Some((prompt_estimate + expected_headroom).try_into().unwrap()),
        None,
    );

    agent.run_turn(prompt, &mut NullUi).await.unwrap();

    assert_eq!(*max_tokens.lock().unwrap(), vec![expected_headroom as u32]);
}

#[tokio::test]
async fn context_preflight_drops_prior_context_before_first_provider_call() {
    let mut cfg = config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.routing.max_tokens = 512;
    cfg.routing.requested_max_tokens = 512;
    let (mut agent, requests) = scripted_agent(
        vec![ProviderStep::Completion(completion(
            vec![Content::Text("ok".into())],
            10,
            2,
        ))],
        cfg,
    );
    let huge_old = "old context ".repeat(20_000);
    agent.messages_mut().push(Message::user(huge_old.clone()));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));

    let prompt = "answer the current question";
    let system_estimate = hi_ai::estimate_messages_tokens(&agent.messages()[..1]);
    let latest_estimate = hi_ai::estimate_text_tokens(prompt);
    let window = system_estimate + latest_estimate + 512 + 128;
    assert!(
        hi_ai::estimate_messages_tokens(agent.messages()) + latest_estimate + 512 > window,
        "test must start over the window before dropping old context"
    );
    agent.set_model("m".into(), Some(window.try_into().unwrap()), None);
    let mut ui = RecordingUi::default();

    agent.run_turn(prompt, &mut ui).await.unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        1,
        "prior context should be dropped before the first provider call"
    );
    let sent = requests[0]
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        sent.contains("Earlier conversation context was omitted"),
        "request includes context omission marker: {sent}"
    );
    assert!(
        sent.contains(prompt),
        "latest user request is preserved: {sent}"
    );
    assert!(
        !sent.contains(&huge_old[..64]),
        "oversized prior context should not be sent"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("dropped prior conversation context")),
        "user sees recovery status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn malformed_stream_retries_and_recovers() {
    // A garbled stream on the first call is silently re-run (with recovery
    // sampling) rather than failing the turn — then it recovers.
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::MalformedStream),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    let mut ui = RecordingUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "retried once after the garble"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("retrying")),
        "shows a retry, got: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn malformed_stream_retry_is_internal_not_user_visible_status() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::MalformedStream),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    let mut ui = SplitUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    assert!(
        ui.statuses.iter().all(|s| !s.contains("retrying")),
        "malformed-stream recovery must not be user-visible status: {:?}",
        ui.statuses
    );
    assert!(
        ui.nudges.iter().any(|s| s.contains("retrying")),
        "internal retry telemetry should remain available to tests: {:?}",
        ui.nudges
    );
}

#[tokio::test]
async fn retry_counts_usage_from_failed_attempt() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::ErrorWithUsage(
                ProviderErrorKind::MalformedStream,
                Usage {
                    input_tokens: 7,
                    output_tokens: 100,
                    ..Default::default()
                },
            ),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(agent.totals().input_tokens, 12);
    assert_eq!(agent.totals().output_tokens, 103);
}

#[tokio::test]
async fn empty_completion_error_is_resampled_too() {
    // The same path catches a provider's empty-completion *error*, not just a
    // content-less Ok response.
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    agent.run_turn("go", &mut NullUi).await.unwrap();
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn empty_completion_after_tool_results_gets_continuation_nudge() {
    let read_cargo = Content::ToolCall {
        id: "r".into(),
        name: "read".into(),
        arguments: serde_json::json!({ "path": "Cargo.toml", "limit": 20 }).to_string(),
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(vec![read_cargo], 5, 1)),
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 8, 2)),
        ],
        config(),
    );

    agent.run_turn("say hi", &mut NullUi).await.unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    let retry_request = &requests[3];
    assert!(
        retry_request
            .last()
            .is_some_and(|message| message.role == Role::User
                && message
                    .text()
                    .contains("previous model response after the tool results was empty")),
        "retry should include a post-tool empty-response nudge: {retry_request:#?}"
    );
    assert!(
        retry_request
            .windows(2)
            .all(|pair| !(pair[0].role == Role::User && pair[1].role == Role::User)),
        "nudge must not create consecutive user messages: {retry_request:#?}"
    );
}

#[tokio::test]
async fn contentless_completion_after_tool_results_gets_continuation_nudge() {
    let read_cargo = Content::ToolCall {
        id: "r".into(),
        name: "read".into(),
        arguments: serde_json::json!({ "path": "Cargo.toml", "limit": 20 }).to_string(),
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(vec![read_cargo], 5, 1)),
            ProviderStep::Completion(completion(vec![], 8, 0)),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 8, 2)),
        ],
        config(),
    );

    agent.run_turn("say hi", &mut NullUi).await.unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let retry_request = &requests[2];
    assert!(
        retry_request
            .last()
            .is_some_and(|message| message.role == Role::User
                && message
                    .text()
                    .contains("previous model response after the tool results was empty")),
        "retry should include a post-tool empty-response nudge: {retry_request:#?}"
    );
}

#[tokio::test]
async fn output_cap_error_retries_once_with_advertised_budget() {
    let mut cfg = config();
    cfg.routing.max_tokens = 8192;
    cfg.routing.requested_max_tokens = 8192;
    let (mut agent, requests, max_tokens) = scripted_agent_recording_max_tokens(
        vec![
            ProviderStep::ErrorMessage(
                ProviderErrorKind::RequestTooLarge,
                "API error 400 Bad Request: max_tokens must be less than or equal to 4096".into(),
            ),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        cfg,
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 2);
    assert_eq!(*max_tokens.lock().unwrap(), vec![8192, 4096]);
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
}

#[tokio::test]
async fn output_cap_error_without_limit_halves_budget_not_2048() {
    let mut cfg = config();
    cfg.routing.max_tokens = 8192;
    cfg.routing.requested_max_tokens = 8192;
    let (mut agent, _requests, max_tokens) = scripted_agent_recording_max_tokens(
        vec![
            ProviderStep::ErrorMessage(
                ProviderErrorKind::UnsupportedRequestShape,
                "max_tokens is greater than the provider output limit".into(),
            ),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        cfg,
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(*max_tokens.lock().unwrap(), vec![8192, 4096]);
}

#[tokio::test]
async fn retryable_route_rejection_retries_and_recovers() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::ErrorMessage(
                ProviderErrorKind::ModelUnavailable,
                r#"API error 503 Service Unavailable: {"error":"model temporarily unavailable","code":"model_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
            ),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 2);
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
}

#[tokio::test]
async fn temporary_provider_overload_gets_extended_retry_budget() {
    let overload = || {
        ProviderStep::ErrorMessage(
            ProviderErrorKind::RateLimit,
            r#"API error 429 Too Many Requests: {"error":{"message":"glm-5.2 is temporarily overloaded","code":1305},"retry_after_seconds":0}"#.into(),
        )
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            overload(),
            overload(),
            overload(),
            overload(),
            overload(),
            overload(),
            overload(),
            overload(),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 9);
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
}

#[tokio::test]
async fn ordinary_rate_limit_retries_with_backoff_budget() {
    let limited = || {
        ProviderStep::ErrorMessage(
            ProviderErrorKind::RateLimit,
            r#"API error 429 Too Many Requests: {"error":{"message":"quota exceeded","code":"rate_limit"},"retry_after_seconds":0}"#.into(),
        )
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            limited(),
            limited(),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 3);
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
}

#[tokio::test]
async fn ordinary_rate_limit_exhausts_backoff_budget() {
    let limited = || {
        ProviderStep::ErrorMessage(
            ProviderErrorKind::RateLimit,
            r#"API error 429 Too Many Requests: {"error":{"message":"too many requests","code":"rate_limit"},"retry_after_seconds":0}"#.into(),
        )
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            limited(),
            limited(),
            limited(),
            limited(),
            limited(),
            limited(),
            limited(),
            limited(),
            limited(),
        ],
        config(),
    );

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::RateLimit)
    );
    // initial attempt + 8 retries
    assert_eq!(requests.lock().unwrap().len(), 9);
}

#[tokio::test]
async fn retryable_route_rejection_exhausts_then_surfaces_error() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::ErrorMessage(
                ProviderErrorKind::ModelUnavailable,
                r#"{"error":"model temporarily unavailable","code":"model_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
            ),
            ProviderStep::ErrorMessage(
                ProviderErrorKind::ModelUnavailable,
                r#"{"error":"model temporarily unavailable","code":"model_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
            ),
            ProviderStep::ErrorMessage(
                ProviderErrorKind::ModelUnavailable,
                r#"{"error":"model temporarily unavailable","code":"model_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
            ),
        ],
        config(),
    );

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::ModelUnavailable)
    );
    assert_eq!(requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn tool_protocol_error_is_resampled_too() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    agent.run_turn("go", &mut NullUi).await.unwrap();
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn tool_protocol_after_tool_progress_gets_guidance_nudge() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(bash_completion("true")),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    agent.run_turn("go", &mut NullUi).await.unwrap();
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(
        requests[2]
            .iter()
            .any(|message| message.text().contains("valid tool calls")),
        "expected protocol guidance in retry request: {:?}",
        requests[2]
    );
}

#[tokio::test]
async fn alternating_invalid_tool_turns_hit_the_cumulative_circuit_breaker() {
    // A model that alternates a valid tool call with an invalid tool turn keeps
    // resetting the *consecutive* protocol counter (MAX_TOOL_PROTOCOL_RETRIES), so
    // without the cumulative cap the nudge-and-retry loop runs forever (the qtest4
    // wedge). The cumulative circuit-breaker must end the turn instead. Distinct
    // valid calls each round keep the repeat-tool-call guard from firing first, so
    // this isolates the protocol cap; far more pairs than the cap are scripted, so
    // a non-terminating loop would exhaust the script and panic in the provider.
    let mut steps = Vec::new();
    for i in 0..16 {
        steps.push(ProviderStep::Completion(bash_completion(&format!(
            "echo {i}"
        ))));
        steps.push(ProviderStep::Error(ProviderErrorKind::ToolProtocol));
    }
    let (mut agent, _requests) = scripted_agent(steps, config());
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    assert!(
        ui.statuses.iter().any(|s| s.contains("invalid tool turns")),
        "the circuit-breaker should end the turn once cumulative invalid turns are spent: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn implementation_tool_protocol_exhaustion_falls_back_to_text_tool_calls() {
    let path = temp_file("protocol-text-fallback");
    let path_string = path.to_string_lossy().to_string();
    let xmlish_write = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>{path_string}</arg_value><arg_key>content</arg_key><arg_value>ok\n</arg_value></tool_call>"
    );
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Completion(completion(vec![Content::Text(xmlish_write)], 5, 3)),
            ProviderStep::Completion(bash_completion("true # validate")),
            ProviderStep::Completion(completion(
                vec![Content::Text(format!(
                    "Changed {path_string} and validated with true # validate."
                ))],
                5,
                3,
            )),
        ],
        config(),
    );
    let mut ui = RecordingUi::default();
    agent
        .run_turn("/build a small CLI project tracker", &mut ui)
        .await
        .unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "ok\n");
    let _ = std::fs::remove_file(&path);

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("plain-text tool-call parsing")),
        "expected text-tool fallback status: {:?}",
        ui.statuses
    );
    assert!(
        agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("validated with true # validate")
    );
    assert!(requests.lock().unwrap().len() >= 7);
}

#[tokio::test]
async fn terminal_error_aborts_without_retry() {
    // A non-resamplable error (auth) fails the turn immediately — no retry.
    let (mut agent, requests) =
        scripted_agent(vec![ProviderStep::Error(ProviderErrorKind::Auth)], config());
    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();
    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        1,
        "a terminal error is not retried"
    );
}

#[tokio::test]
async fn terminal_error_resets_stale_turn_telemetry() {
    let (mut agent, _requests) =
        scripted_agent(vec![ProviderStep::Error(ProviderErrorKind::Auth)], config());
    agent.last_turn_telemetry = TurnTelemetry {
        repeat_nudges: 99,
        stalled_unfinished: true,
        tool_calls: 42,
        ..TurnTelemetry::default()
    };

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.repeat_nudges, 0);
    assert_eq!(telemetry.tool_calls, 0);
    assert!(
        !telemetry.stalled_unfinished,
        "terminal error should report this failed turn, not stale prior telemetry"
    );
}

#[tokio::test]
async fn terminal_error_after_recovery_retry_reports_retry_count() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::MalformedStream),
            ProviderStep::Error(ProviderErrorKind::Auth),
        ],
        config(),
    );

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );
    assert_eq!(
        agent.last_turn_telemetry().recovery_retries,
        1,
        "retry telemetry must survive a terminal error after recovery sampling"
    );
}

#[tokio::test]
async fn terminal_error_after_tool_progress_reports_changed_files_and_tools() {
    let workspace = IsolatedWorkspace::new("retry-terminal-error-progress");
    let path = workspace.path("changed.rs");
    let path_string = path.to_string_lossy().to_string();
    let file_name = path.file_name().unwrap().to_string_lossy().to_string();
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_completion(&path_string)),
            ProviderStep::Error(ProviderErrorKind::Auth),
        ],
        workspace.config(),
    );

    let err = agent
        .run_turn("write the file then continue", &mut NullUi)
        .await
        .unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );
    assert!(
        agent
            .last_changed_files()
            .iter()
            .any(|changed| changed == &file_name),
        "changed file should be retained after terminal error: {:?}",
        agent.last_changed_files()
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.tool_calls, 1);
    assert!(
        telemetry
            .tool_timeline
            .iter()
            .any(|entry| entry.tool == "write" && entry.path == path_string),
        "write tool telemetry should be retained after terminal error: {:?}",
        telemetry.tool_timeline
    );
}

#[tokio::test]
async fn terminal_error_drops_failed_prompt_before_next_turn() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::Auth),
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 1, 1)),
        ],
        config(),
    );

    let err = agent.run_turn("first task", &mut NullUi).await.unwrap_err();
    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );

    agent.run_turn("second task", &mut NullUi).await.unwrap();
    let last_user = agent
        .messages()
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .expect("user prompt recorded");
    let text = last_user.text();
    assert!(
        !text.contains("first task"),
        "failed prompt should not fold into the next turn: {text}"
    );
    assert!(
        text.contains("second task"),
        "next prompt should be cleanly recorded: {text}"
    );
}

#[tokio::test]
async fn protocol_retry_terminal_error_drops_retry_guidance_before_next_turn() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::Auth),
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 1, 1)),
        ],
        config(),
    );

    let err = agent.run_turn("first task", &mut NullUi).await.unwrap_err();
    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );

    agent.run_turn("second task", &mut NullUi).await.unwrap();
    let last_user = agent
        .messages()
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .expect("user prompt recorded");
    let text = last_user.text();
    assert!(
        !text.contains("valid tool calls") && !text.contains("[hi:nudge:"),
        "retry guidance should not leak into the next turn: {text}"
    );
    assert!(
        !text.contains("first task") && text.contains("second task"),
        "next prompt should be cleanly recorded: {text}"
    );
}

#[tokio::test]
async fn terminal_error_persists_usage_before_returning() {
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let (mut agent, _requests) = scripted_agent(
        vec![ProviderStep::ErrorWithUsage(
            ProviderErrorKind::Outage,
            Usage {
                input_tokens: 11,
                output_tokens: 100,
                ..Default::default()
            },
        )],
        config(),
    );
    agent.set_session(Box::new(RecordingSession {
        records: records.clone(),
    }));

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Outage)
    );
    assert_eq!(
        *records.lock().unwrap(),
        vec![Usage {
            input_tokens: 11,
            output_tokens: 100,
            ..Default::default()
        }]
    );
}
