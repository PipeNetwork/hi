use super::*;
use super::common::*;

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
            ProviderStep::Completion(bash_completion("cargo test --help")),
            ProviderStep::Completion(completion(
                vec![Content::Text(format!(
                    "Changed {path_string} and validated with cargo test --help."
                ))],
                5,
                3,
            )),
        ],
        config(),
    );
    let mut ui = RecordingUi::default();
    agent
        .run_turn("build a small CLI GPU training time estimator", &mut ui)
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
            .contains("validated with cargo test --help")
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
        vec![(
            Usage {
                input_tokens: 11,
                output_tokens: 100,
                ..Default::default()
            },
            None,
        )]
    );
}

