use super::common::*;
use super::*;
use std::sync::Arc;

type DecisionRecords = Arc<Mutex<Vec<Vec<Decision>>>>;

struct DecisionRecordingSession {
    records: DecisionRecords,
}

impl SessionSink for DecisionRecordingSession {
    fn record(
        &mut self,
        _messages: &[Message],
        _usage: Usage,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_decisions(&mut self, decisions: &DecisionLog) -> anyhow::Result<()> {
        self.records
            .lock()
            .unwrap()
            .push(decisions.entries().to_vec());
        Ok(())
    }
}

struct FailingDecisionSession;

impl SessionSink for FailingDecisionSession {
    fn record(
        &mut self,
        _messages: &[Message],
        _usage: Usage,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_decisions(&mut self, _decisions: &DecisionLog) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("disk full"))
    }
}

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

#[test]
fn resume_restores_decision_log_and_rebuilds_system_prompt() {
    let mut decisions = DecisionLog::default();
    decisions.record(Decision {
        summary: "use BTreeMap".into(),
        rationale: "ordered iteration".into(),
        files: vec!["src/m.rs".into()],
    });
    let agent = Agent::resume(
        Box::new(Canned(Mutex::new(Vec::new()))),
        config(),
        vec![Message::system("old prompt without decisions")],
        Usage::default(),
        Vec::new(),
        None,
        decisions,
    );

    assert_eq!(agent.decisions().entries().len(), 1);
    assert_eq!(agent.decisions().entries()[0].summary, "use BTreeMap");
    let sys = agent.messages()[0].text();
    assert!(
        sys.contains("use BTreeMap") && sys.contains("ordered iteration"),
        "resume should inject persisted decisions into rebuilt system prompt: {sys}"
    );
    assert!(
        !sys.contains("old prompt without decisions"),
        "resume should rebuild, not keep stale prompt text: {sys}"
    );
}

#[test]
fn record_decision_persists_log_before_updating_visible_prompt() {
    let records = Arc::new(Mutex::new(Vec::new()));
    let mut agent = agent(vec![], config());
    agent.set_session(Box::new(DecisionRecordingSession {
        records: records.clone(),
    }));

    let result = agent.handle_record_decision(
        r#"{"summary":"use BTreeMap","rationale":"ordered iteration","files":["src/m.rs"]}"#,
    );

    assert!(result.contains("Decision recorded"), "result: {result}");
    assert_eq!(agent.decisions().entries().len(), 1);
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0][0].summary, "use BTreeMap");
    assert!(agent.messages()[0].text().contains("use BTreeMap"));
}

#[test]
fn record_decision_keeps_visible_prompt_unchanged_when_persistence_fails() {
    let mut agent = agent(vec![], config());
    agent.set_session(Box::new(FailingDecisionSession));

    let result = agent.handle_record_decision(
        r#"{"summary":"use BTreeMap","rationale":"ordered iteration","files":["src/m.rs"]}"#,
    );

    assert!(
        result.contains("couldn't persist decision"),
        "result: {result}"
    );
    assert!(agent.decisions().entries().is_empty());
    assert!(!agent.messages()[0].text().contains("use BTreeMap"));
}
