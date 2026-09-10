use super::*;

#[tokio::test]
async fn consecutive_protocol_retries_replace_the_format_nudge() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    agent.run_turn("go", &mut NullUi).await.unwrap();
    let protocol_hits = agent
        .messages()
        .iter()
        .map(|message| message.text().matches("[hi:nudge:protocol]").count())
        .sum::<usize>();
    assert_eq!(
        protocol_hits,
        1,
        "protocol retries must replace the format reminder, not stack copies: {:?}",
        agent
            .messages()
            .iter()
            .map(|message| message.text())
            .collect::<Vec<_>>()
    );
}
