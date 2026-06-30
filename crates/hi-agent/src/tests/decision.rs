use super::*;
use super::common::*;

#[tokio::test]
async fn record_decision_persists_across_compaction_in_system_prompt() {
    // A decision recorded via the tool survives a compaction in the system
    // prompt (the log is injected into the system message, which compaction
    // preserves verbatim — not summarized away).
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "d1".into(),
                name: "record_decision".into(),
                arguments: r#"{"summary":"use BTreeMap","rationale":"ordered iteration","files":["src/m.rs"]}"#.into(),
            }],
            1,
            1,
        ),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    agent.run_turn("refactor", &mut NullUi).await.unwrap();
    assert_eq!(agent.decisions().entries().len(), 1);
    assert_eq!(agent.decisions().entries()[0].summary, "use BTreeMap");

    // The system prompt contains the decision.
    let sys = agent.messages()[0].text();
    assert!(
        sys.contains("use BTreeMap") && sys.contains("ordered iteration"),
        "decision in system prompt: {sys}"
    );

    // A compaction that summarizes the Q&A tail must NOT remove the
    // decision from the system prompt.
    agent
        .compact_with(CompactionKind::Summarize, &mut NullUi)
        .await
        .unwrap();
    let sys_after = agent.messages()[0].text();
    assert!(
        sys_after.contains("use BTreeMap"),
        "decision survives compaction: {sys_after}"
    );
}

