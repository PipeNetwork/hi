use super::common::*;
use super::*;

#[tokio::test]
async fn compact_replaces_history_with_summary() {
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let responses = vec![completion(
        vec![Content::Text(
            "BRIEF: ported the parser; tests green".into(),
        )],
        7,
        5,
    )];
    let mut agent = agent(responses, config());
    agent.set_session(Box::new(RecordingSession {
        records: records.clone(),
    }));
    // Some history to compact.
    agent.messages_mut().push(Message::user("hello"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("hi".into())]));

    agent.compact(&mut NullUi).await.unwrap();

    // History collapses to system + summary.
    assert_eq!(agent.messages().len(), 2);
    assert_eq!(agent.messages()[0].role, Role::System);
    assert!(
        agent.messages()[1]
            .text()
            .contains("BRIEF: ported the parser"),
        "summary message retained"
    );
    // The summarization call's usage is counted.
    assert_eq!(agent.totals().output_tokens, 5);
    assert_eq!(
        *records.lock().unwrap(),
        vec![Usage {
            input_tokens: 7,
            output_tokens: 5,
            ..Default::default()
        }],
        "manual compaction persists summarization usage before writing the durable boundary"
    );
}

#[tokio::test]
async fn hybrid_keeps_recent_and_folds_summary() {
    let mut agent = agent(
        vec![completion(vec![Content::Text("OLD SUMMARY".into())], 3, 2)],
        config(),
    );
    // Two user turns; keep_recent = 1 summarizes the first, keeps the second.
    agent.messages_mut().push(Message::user("q1"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a1".into())]));
    agent.messages_mut().push(Message::user("q2"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a2".into())]));

    agent
        .compact_with(CompactionKind::Hybrid { keep_recent: 1 }, &mut NullUi)
        .await
        .unwrap();

    let m = agent.messages();
    // system + (summary folded into kept user turn) + kept assistant reply.
    assert_eq!(m.len(), 3);
    assert_eq!(m[0].role, Role::System);
    assert_eq!(m[1].role, Role::User);
    assert!(
        m[1].text().contains("OLD SUMMARY"),
        "summary folded: {}",
        m[1].text()
    );
    assert!(
        m[1].text().contains("q2"),
        "recent turn kept: {}",
        m[1].text()
    );
    assert_eq!(m[2].text(), "a2");
    // No two consecutive same-role messages (provider-safe).
    assert!(
        m.windows(2).all(|w| w[0].role != w[1].role),
        "roles must alternate"
    );
}

#[tokio::test]
async fn hybrid_keep_recent_zero_summarizes_instead_of_panicking() {
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("WHOLE SUMMARY".into())],
            3,
            2,
        )],
        config(),
    );
    agent.messages_mut().push(Message::user("q1"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a1".into())]));
    agent.messages_mut().push(Message::user("q2"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a2".into())]));

    agent
        .compact_with(CompactionKind::Hybrid { keep_recent: 0 }, &mut NullUi)
        .await
        .unwrap();

    let m = agent.messages();
    assert_eq!(m.len(), 2);
    assert_eq!(m[0].role, Role::System);
    assert!(m[1].text().contains("WHOLE SUMMARY"));
    agent.messages.validate_for_provider().unwrap();
}

#[tokio::test]
async fn elide_then_summarize_tail_elides_tool_turns_summarizes_qa() {
    // A session with: an old tool-bearing turn (q1 + read + big result), an
    // old Q&A turn (q2 + text), and a recent turn (q3). The new default
    // strategy should elide the old tool result (keep the call/result
    // skeleton) and summarize only the old Q&A tail, folding the summary
    // into the first kept turn. The recent turn stays verbatim.
    let mut agent = agent(
        vec![completion(vec![Content::Text("QA SUMMARY".into())], 1, 1)],
        config(),
    );
    // Old tool-bearing turn.
    agent.messages_mut().push(Message::user("q1"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("c1", "x".repeat(500)));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "a1 after reading the file".into(),
        )]));
    // Old Q&A turn (no tool results) — this is the conversational tail.
    agent.messages_mut().push(Message::user("q2"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a2".into())]));
    // Recent turn.
    agent.messages_mut().push(Message::user("q3"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a3".into())]));

    agent
        .compact_with(
            CompactionKind::ElideThenSummarizeTail { keep_recent: 1 },
            &mut NullUi,
        )
        .await
        .unwrap();

    let m = agent.messages();
    // The old tool result must be elided (skeleton kept, not wiped).
    let tool_results: Vec<&str> = m
        .iter()
        .flat_map(|msg| &msg.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        tool_results.iter().any(|o| o.starts_with("[elided")),
        "old tool result elided (skeleton kept): {tool_results:?}"
    );
    assert!(
        !tool_results.iter().any(|o| o.contains(&"x".repeat(100))),
        "old tool output content gone: {tool_results:?}"
    );
    assert!(
        m.iter()
            .any(|msg| msg.role == Role::User && msg.text().contains("q1")),
        "old tool-bearing user prompt preserved: {m:?}"
    );
    assert!(
        m.iter().any(|msg| msg.text().contains("a1 after reading")),
        "old post-tool assistant answer preserved: {m:?}"
    );
    // The Q&A summary is folded into the first kept turn (q3), and q3 stays.
    let user_texts: Vec<String> = m
        .iter()
        .filter(|msg| msg.role == Role::User)
        .map(|msg| msg.text())
        .collect();
    assert!(
        user_texts.iter().any(|t| t.contains("QA SUMMARY")),
        "Q&A tail summarized and folded: {user_texts:?}"
    );
    assert!(
        user_texts.iter().any(|t| t.contains("q3")),
        "recent turn kept: {user_texts:?}"
    );
    assert!(
        !user_texts.iter().any(|t| t == "q2"),
        "old Q&A prompt was summarized, not kept verbatim: {user_texts:?}"
    );
    // Provider-safe: roles alternate.
    assert!(
        m.windows(2).all(|w| w[0].role != w[1].role),
        "roles must alternate: {:?}",
        m.iter().map(|x| x.role).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn elide_then_summarize_tail_keep_recent_zero_summarizes_instead_of_panicking() {
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("WHOLE SUMMARY".into())],
            3,
            2,
        )],
        config(),
    );
    agent.messages_mut().push(Message::user("q1"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("c1", "x".repeat(500)));
    agent.messages_mut().push(Message::user("q2"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a2".into())]));

    agent
        .compact_with(
            CompactionKind::ElideThenSummarizeTail { keep_recent: 0 },
            &mut NullUi,
        )
        .await
        .unwrap();

    let m = agent.messages();
    assert_eq!(m.len(), 2);
    assert_eq!(m[0].role, Role::System);
    assert!(m[1].text().contains("WHOLE SUMMARY"));
    agent.messages.validate_for_provider().unwrap();
}

#[tokio::test]
async fn elide_then_summarize_tail_skips_model_call_when_no_qa_tail() {
    // A pure tool-heavy session (no old Q&A turns): the strategy should
    // elide and NOT make a summarizing model call. Provide no canned
    // completion — if it tried to summarize, the provider would panic on
    // an empty response list.
    let mut agent = agent(vec![], config());
    agent.messages_mut().push(Message::user("q1"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("c1", "x".repeat(500)));
    agent.messages_mut().push(Message::user("q2"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a2".into())]));

    // keep_recent = 1 → q2 is recent; q1's tool result is old and gets
    // elided. No Q&A tail older than q2 → no model call.
    agent
        .compact_with(
            CompactionKind::ElideThenSummarizeTail { keep_recent: 1 },
            &mut NullUi,
        )
        .await
        .unwrap();
    let m = agent.messages();
    let tool_results: Vec<&str> = m
        .iter()
        .flat_map(|msg| &msg.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        tool_results.iter().any(|o| o.starts_with("[elided")),
        "old tool result elided: {tool_results:?}"
    );
}

#[tokio::test]
async fn hybrid_falls_back_to_summarize_when_too_few_turns() {
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("WHOLE SUMMARY".into())],
            1,
            1,
        )],
        config(),
    );
    agent.messages_mut().push(Message::user("only turn"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a".into())]));
    // keep_recent = 3 but only one turn → no recent window → summarize all.
    agent
        .compact_with(CompactionKind::Hybrid { keep_recent: 3 }, &mut NullUi)
        .await
        .unwrap();
    let m = agent.messages();
    assert_eq!(m.len(), 2);
    assert_eq!(m[0].role, Role::System);
    assert!(m[1].text().contains("WHOLE SUMMARY"));
}

#[tokio::test]
async fn elide_shrinks_old_tool_output_without_a_model_call() {
    // Empty provider: if elision tried to call the model, this would panic.
    let mut agent = agent(vec![], config());
    let big = "x".repeat(500);
    agent.messages_mut().push(Message::user("read a"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("c1", big.clone()));
    agent.messages_mut().push(Message::user("read b")); // recent turn
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "c2".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("c2", big.clone()));

    agent
        .compact_with(
            CompactionKind::ElideToolOutput { keep_recent: 1 },
            &mut NullUi,
        )
        .await
        .unwrap();

    let outputs: Vec<String> = agent
        .messages()
        .iter()
        .flat_map(|m| &m.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect();
    assert!(
        outputs[0].starts_with("[elided"),
        "old elided: {}",
        outputs[0]
    );
    assert_eq!(outputs[1], big, "recent kept verbatim");
}
