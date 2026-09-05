use super::common::*;
use super::*;
use std::sync::Arc;

fn add_active_turn(agent: &mut Agent) -> usize {
    agent.messages_mut().extend([
        Message::user("old task"),
        Message::assistant(vec![Content::Text("old answer".into())]),
    ]);
    let start = agent.messages().len();
    agent.messages_mut().extend([
        Message {
            role: Role::User,
            content: vec![
                Content::Text("Match this screenshot".into()),
                Content::Image {
                    data: "original-screenshot".into(),
                    media_type: "image/png".into(),
                },
                Content::Text("Keep the original footer".into()),
            ],
        },
        Message::assistant(vec![Content::Text("I will inspect the layout".into())]),
        Message {
            role: Role::User,
            content: vec![
                Content::Text("[user-message]\nUse the blue header instead".into()),
                Content::Image {
                    data: "corrected-header".into(),
                    media_type: "image/png".into(),
                },
            ],
        },
        Message::assistant(vec![Content::Text("Header correction noted".into())]),
        Message::user("[hi:nudge:continue]\nContinue this task"),
    ]);
    start
}

fn assert_active_request(messages: &[Message]) {
    let text = messages
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    for required in [
        "Match this screenshot",
        "Keep the original footer",
        "Use the blue header instead",
        "Continue this task",
    ] {
        assert!(text.contains(required), "recovery lost {required}: {text}");
    }
    for required in ["original-screenshot", "corrected-header"] {
        assert!(
            messages
                .iter()
                .flat_map(|message| &message.content)
                .any(|block| matches!(block, Content::Image { data, .. } if data == required)),
            "recovery lost image {required}"
        );
    }
}

#[test]
fn overflow_compaction_preserves_original_request_before_correction_and_nudge() {
    let mut agent = agent(vec![], config());
    let start = add_active_turn(&mut agent);
    assert!(
        agent
            .retry_after_request_too_large_compact(start, &mut NullUi)
            .unwrap()
    );
    assert_active_request(agent.messages());
    agent.messages.validate_for_provider().unwrap();
}

#[test]
fn overflow_drop_preserves_typed_request_and_later_corrections() {
    let mut agent = agent(vec![], config());
    let start = add_active_turn(&mut agent);
    assert!(
        agent
            .retry_after_request_too_large("Match this screenshot", start, &mut NullUi)
            .unwrap()
    );
    assert_active_request(agent.messages());
    assert_eq!(agent.messages().len(), 2);
    agent.messages.validate_for_provider().unwrap();
}

#[test]
fn overflow_drop_after_compaction_retains_all_current_user_messages() {
    let mut agent = agent(vec![], config());
    let start = add_active_turn(&mut agent);
    assert!(
        agent
            .retry_after_request_too_large_compact(start, &mut NullUi)
            .unwrap()
    );
    assert!(
        agent
            .retry_after_request_too_large("Match this screenshot", 1, &mut NullUi)
            .unwrap()
    );
    assert_active_request(agent.messages());
    assert!(
        !agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains(COMPACTION_REFERENCE_PREFIX)
    );
    agent.messages.validate_for_provider().unwrap();
}

#[test]
fn context_preflight_preserves_typed_request_and_later_corrections() {
    let mut cfg = config();
    cfg.routing.context_window = Some(200_000);
    cfg.memory.auto_compact = false;
    let mut agent = agent(vec![], cfg);
    let start = add_active_turn(&mut agent);
    agent.messages_mut()[1] = Message::user("old context ".repeat(100_000));
    let result = agent
        .ensure_request_fits_context(
            "Match this screenshot",
            start,
            100,
            0,
            crate::agent::ContextWindowLimits::default(),
            &mut NullUi,
        )
        .unwrap();
    assert!(result.dropped_prior_context);
    assert_active_request(agent.messages());
    agent.messages.validate_for_provider().unwrap();
}

struct RecoverySession(Arc<Mutex<Vec<Vec<Message>>>>);

impl SessionSink for RecoverySession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }
    fn record_compaction(&mut self, messages: &[Message]) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(messages.to_vec());
        Ok(())
    }
}

#[test]
fn overflow_drop_persists_recovered_request_with_replacement_boundary() {
    let mut agent = agent(vec![], config());
    let start = add_active_turn(&mut agent);
    let records = Arc::new(Mutex::new(Vec::new()));
    agent.set_session(Box::new(RecoverySession(records.clone())));
    assert!(
        agent
            .retry_after_request_too_large("Match this screenshot", start, &mut NullUi)
            .unwrap()
    );
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_active_request(&records[0]);
    assert_eq!(
        serde_json::to_string(&records[0]).unwrap(),
        serde_json::to_string(agent.messages()).unwrap()
    );
}

#[tokio::test]
async fn provider_retries_keep_image_after_both_overflow_recovery_stages() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::RequestTooLarge,
            ProviderStep::RequestTooLarge,
            ProviderStep::Completion(completion(
                vec![Content::Text("The header is blue".into())],
                12,
                3,
            )),
        ],
        config(),
    );
    agent.messages_mut().extend([
        Message::user("old task"),
        Message::assistant(vec![Content::Text("old answer".into())]),
        Message::user("Pending request before the session was resumed"),
    ]);
    agent
        .run_prompt(
            hi_ai::PromptInput::text("What color is the header?").image("screenshot", "image/png"),
            &mut NullUi,
        )
        .await
        .unwrap();
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    for (index, request) in requests.iter().enumerate() {
        assert!(
            request.iter().any(|message| message
                .text()
                .contains("Pending request before the session was resumed")),
            "request {index} lost the pending user request"
        );
        assert!(
            request
                .iter()
                .flat_map(|message| &message.content)
                .any(|block| matches!(block, Content::Image { data, .. } if data == "screenshot")),
            "request {index} lost the screenshot"
        );
    }
}

#[tokio::test]
async fn failed_preflight_after_context_drop_removes_oversized_request() {
    let mut cfg = config();
    cfg.routing.context_window = Some(5_000);
    cfg.memory.auto_compact = false;
    let (mut agent, requests) = scripted_agent(vec![], cfg);
    agent.messages_mut().extend([
        Message::user("old task"),
        Message::assistant(vec![Content::Text("old answer".into())]),
    ]);
    let error = agent
        .run_turn(&"oversized current request ".repeat(10_000), &mut NullUi)
        .await
        .unwrap_err();
    assert_eq!(
        hi_ai::provider_error_kind(&error),
        Some(ProviderErrorKind::RequestTooLarge)
    );
    assert!(requests.lock().unwrap().is_empty());
    assert_eq!(
        agent.messages().len(),
        1,
        "failed preflight retained its oversized rewritten request after the history boundary moved"
    );
}
