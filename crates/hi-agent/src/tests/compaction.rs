use super::common::*;
use super::*;
struct RejectingCompactionSession;

impl SessionSink for RejectingCompactionSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        anyhow::bail!("injected compaction boundary failure")
    }
}

struct PendingCompactionProvider {
    entered: std::sync::Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl Provider for PendingCompactionProvider {
    async fn stream(
        &self,
        _request: hi_ai::ChatRequest,
        _sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        self.entered.notify_one();
        std::future::pending().await
    }
}

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

    let context_generation = agent.runtime.context_generation();
    agent.compact(&mut NullUi).await.unwrap();

    // History collapses to system + summary.
    assert_eq!(
        agent.runtime.context_generation(),
        context_generation + 1,
        "a completed compaction must invalidate the active context generation"
    );
    assert_eq!(agent.messages().len(), 2);
    assert_eq!(agent.messages()[0].role, Role::System);
    assert!(
        agent.messages()[1]
            .text()
            .contains("BRIEF: ported the parser"),
        "summary message retained"
    );
    assert!(
        agent.messages()[1].text().contains("REFERENCE ONLY"),
        "compacted summary is marked as historical reference: {}",
        agent.messages()[1].text()
    );
    assert!(
        agent.messages()[1]
            .text()
            .contains("latest user message wins"),
        "summary boundary tells the model the latest prompt wins: {}",
        agent.messages()[1].text()
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
async fn compaction_is_a_read_only_unified_job_sealed_after_publication() {
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("durable summary".into())],
            2,
            1,
        )],
        config(),
    );
    agent.messages_mut().push(Message::user("old task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    let state_root = agent.state_root().to_path_buf();
    let binding = agent.workspace_controller_binding();

    agent
        .compact_with(CompactionKind::Summarize, &mut NullUi)
        .await
        .unwrap();

    let store = hi_control::ControlStore::open_for_state(state_root).unwrap();
    let jobs = store.jobs_for_binding(binding.binding_id.as_str()).unwrap();
    let compaction = jobs
        .iter()
        .find(|job| job.kind == hi_control::ControlJobKind::Compaction)
        .expect("compaction job projection");
    assert_eq!(
        compaction.effect_scope,
        hi_control::ControlEffectScope::ReadOnly
    );
    assert_eq!(compaction.state, hi_control::ControlJobState::Succeeded);
    assert!(agent.workspace_controller_status().active_jobs.is_empty());
    assert!(agent.messages()[1].text().contains("durable summary"));
}

#[tokio::test]
async fn compaction_boundary_failure_leaves_live_transcript_untouched() {
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("unpublished summary".into())],
            2,
            1,
        )],
        config(),
    );
    agent
        .messages_mut()
        .push(Message::user("keep this exact task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "keep this answer".into(),
        )]));
    let before = serde_json::to_value(agent.messages()).unwrap();
    let state_root = agent.state_root().to_path_buf();
    let binding = agent.workspace_controller_binding();
    agent.set_session(Box::new(RejectingCompactionSession));

    let error = agent
        .compact_with(CompactionKind::Summarize, &mut NullUi)
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("speculative compaction boundary")
    );
    assert_eq!(serde_json::to_value(agent.messages()).unwrap(), before);
    let store = hi_control::ControlStore::open_for_state(state_root).unwrap();
    let jobs = store.jobs_for_binding(binding.binding_id.as_str()).unwrap();
    let compaction = jobs
        .iter()
        .find(|job| job.kind == hi_control::ControlJobKind::Compaction)
        .expect("failed compaction job projection");
    assert_eq!(compaction.state, hi_control::ControlJobState::Failed);
}

#[tokio::test]
async fn cancelled_compaction_seals_its_read_only_job() {
    let entered = std::sync::Arc::new(tokio::sync::Notify::new());
    let config = config();
    let _state_guard = config.test_state_root.clone();
    let mut agent = Agent::new(
        std::sync::Arc::new(PendingCompactionProvider {
            entered: entered.clone(),
        }),
        config,
    )
    .unwrap();
    agent.messages_mut().push(Message::user("old task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    let state_root = agent.state_root().to_path_buf();
    let binding = agent.workspace_controller_binding();
    let worker = tokio::spawn(async move {
        agent
            .compact_with(CompactionKind::Summarize, &mut NullUi)
            .await
    });
    entered.notified().await;
    worker.abort();
    assert!(worker.await.unwrap_err().is_cancelled());

    let settled = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let store = hi_control::ControlStore::open_for_state(&state_root).unwrap();
            let jobs = store.jobs_for_binding(binding.binding_id.as_str()).unwrap();
            if jobs.iter().any(|job| {
                job.kind == hi_control::ControlJobKind::Compaction
                    && job.state == hi_control::ControlJobState::Cancelled
            }) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    if settled.is_err() {
        let store = hi_control::ControlStore::open_for_state(&state_root).unwrap();
        let jobs = store.jobs_for_binding(binding.binding_id.as_str()).unwrap();
        panic!("cancelled compaction lifecycle callback timed out: {jobs:?}");
    }
}

#[tokio::test]
async fn compact_does_not_summarize_stale_turn_controls() {
    let (mut agent, requests) = scripted_agent(
        vec![ProviderStep::Completion(completion(
            vec![Content::Text("clean summary".into())],
            1,
            1,
        ))],
        config(),
    );
    let poisoned = format!(
        "{}\nold context\n{}\n\nYou are in PLAN MODE. Do not modify files or run mutating commands.\nProduce a clear plan and wait.\n\nUser request:\nbuild profiles\n\nRead-only review guard: shell execution (`bash`) and mutation tools are unavailable for this review. Use only the advertised read-only inspection tools; do not write.",
        crate::transcript::CONTEXT_BLOCK_START,
        crate::transcript::CONTEXT_BLOCK_END,
    );
    agent.messages_mut().push(Message::user(poisoned));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "I'm in plan mode; the plan is ready.".into(),
        )]));
    agent
        .compact_with(CompactionKind::Summarize, &mut NullUi)
        .await
        .unwrap();

    let request_text = requests
        .lock()
        .unwrap()
        .first()
        .expect("summarizer request")
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(request_text.contains("build profiles"));
    assert!(!request_text.contains("Plan mode is ON for this turn"));
    assert!(!request_text.contains("Read-only review guard:"));
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
    assert!(
        m[1].text().contains("--- LATEST USER MESSAGE ---"),
        "folded summary is separated from active user text: {}",
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

#[tokio::test]
async fn read_only_safety_window_preserves_history_within_real_window() {
    // Regression: a read-only turn passes a 12k SOFT safety window into
    // ensure_request_fits_context. That must only drive non-destructive elision —
    // it must NOT be treated as the real context window and durably discard the
    // whole session. Here the real window is 200k and the estimate (~15k tokens)
    // is over the 12k soft preference but far under 200k, so history stays.
    let mut cfg = config();
    cfg.routing.context_window = Some(200_000);
    cfg.memory.auto_compact = false;
    let mut agent = agent(vec![], cfg);
    agent.messages_mut().push(Message::user("x".repeat(60_000))); // ~15k tokens
    let before = agent.messages().len();
    let pre = agent
        .ensure_request_fits_context(
            "review this module",
            2,
            100,
            0,
            crate::agent::ContextWindowLimits {
                safety: Some(12_000),
                provider: None,
            },
            &mut NullUi,
        )
        .expect("must not hard-fail within the real window");
    assert!(
        !pre.dropped_prior_context,
        "read-only safety window must not drop prior history"
    );
    assert_eq!(
        agent.messages().len(),
        before,
        "no messages may be discarded"
    );
}

#[tokio::test]
async fn context_preflight_still_drops_when_real_window_exceeded() {
    // The destructive drop must still fire when the REAL model window is
    // genuinely exceeded (turn_start > 1), so the fix above doesn't disable
    // legitimate overflow recovery.
    let mut cfg = config();
    cfg.routing.context_window = Some(200_000);
    cfg.memory.auto_compact = false;
    let mut agent = agent(vec![], cfg);
    agent
        .messages_mut()
        .push(Message::user("y".repeat(1_000_000))); // ~250k tokens
    let pre = agent
        .ensure_request_fits_context(
            "continue",
            2,
            100,
            0,
            crate::agent::ContextWindowLimits::default(),
            &mut NullUi,
        )
        .expect("dropping history brings the request under the window");
    assert!(
        pre.dropped_prior_context,
        "genuine overflow must still drop prior context"
    );
}

#[test]
fn in_turn_elide_runs_without_a_catalog_window() {
    // Mutation turns used to no-op elision when `/models` omitted a window.
    // Occupancy must still stub old bulky tool results against the fallback
    // window so a long loop does not resend every payload.
    let mut cfg = config();
    cfg.routing.context_window = None;
    cfg.memory.auto_compact = true;
    cfg.memory.in_turn_keep_tool_results = 1;
    cfg.memory.in_turn_elide_percent = 1;
    let mut agent = agent(vec![], cfg);
    for index in 0..4 {
        let id = format!("c{index}");
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
        agent
            .messages_mut()
            .push(Message::tool_result(&id, "x".repeat(8_000)));
    }
    agent.elide_in_turn_context_if_needed(&mut NullUi, None);
    let outputs: Vec<&str> = agent
        .messages()
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            Content::ToolResult { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        outputs.iter().any(|output| output.starts_with("[elided")),
        "old tool results must be stubbed without a catalog window: {outputs:?}"
    );
    assert!(
        outputs.last().is_some_and(|output| output.len() == 8_000),
        "newest tool result stays verbatim: {outputs:?}"
    );
}

#[test]
fn persist_after_transcript_shrink_does_not_panic() {
    // Regression: strip_trailing_nudges/strip_finalize_pair pop messages without
    // moving the `persisted` cursor, so after a mid-turn persist the cursor can
    // exceed the length. persist() must clamp rather than slice out of bounds.
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let mut agent = agent(vec![], config());
    agent.set_session(Box::new(RecordingSession {
        records: records.clone(),
    }));
    agent.messages_mut().push(Message::user("a"));
    agent.messages_mut().push(Message::user("b"));
    agent.persist().unwrap(); // persisted == 2
    agent.messages_mut().pop(); // len == 1, persisted == 2 (> len)
    agent.persist().unwrap(); // must not panic on the [persisted..] slice
}
