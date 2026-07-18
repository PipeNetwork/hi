use super::common::*;
use super::*;

#[tokio::test]
async fn scheduler_parallelism_counts_concurrent_batches() {
    // A batch of independent reads (different paths, no deps) should run
    // concurrently — telemetry reports max_concurrent_batch > 1 and a
    // sub-100% serial share. Pins that the dep-aware scheduler's
    // concurrency is measured, not just shipped on faith.
    let cfg = config();
    let responses = vec![
        completion(
            vec![
                Content::ToolCall {
                    id: "r1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"a.rs"}"#.into(),
                },
                Content::ToolCall {
                    id: "r2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"b.rs"}"#.into(),
                },
                Content::ToolCall {
                    id: "r3".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"c.rs"}"#.into(),
                },
            ],
            1,
            1,
        ),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("read them", &mut ui).await.unwrap();
    let tel = agent.last_turn_telemetry();
    assert_eq!(tel.tool_calls, 3, "three reads ran: {:?}", tel);
    assert!(
        tel.max_concurrent_batch >= 2,
        "independent reads overlapped: {:?}",
        tel
    );
    assert!(
        tel.serial_runs < tel.tool_calls,
        "not all serial: {:?}",
        tel
    );
    // The timeline records each call with its tool name and path.
    assert_eq!(
        tel.tool_timeline.len(),
        3,
        "timeline has one entry per call: {:?}",
        tel.tool_timeline
    );
    let tools: Vec<&str> = tel.tool_timeline.iter().map(|e| e.tool.as_str()).collect();
    assert!(tools.iter().all(|&t| t == "read"), "all reads: {tools:?}");
    let paths: Vec<&str> = tel.tool_timeline.iter().map(|e| e.path.as_str()).collect();
    assert!(
        paths.contains(&"a.rs") && paths.contains(&"b.rs") && paths.contains(&"c.rs"),
        "timeline paths match calls: {paths:?}"
    );
    assert!(
        tel.tool_timeline.iter().all(|e| e.error),
        "reads error (files don't exist in test): {:?}",
        tel.tool_timeline
    );
}

#[tokio::test]
async fn hard_tool_budget_reserves_only_the_model_ordered_prefix() {
    let mut cfg = config();
    cfg.max_tool_calls = 2;
    let calls = (1..=3)
        .map(|index| Content::ToolCall {
            id: format!("r{index}"),
            name: "read".into(),
            arguments: format!(r#"{{"path":"{index}.rs"}}"#),
        })
        .collect();
    let responses = vec![
        completion(calls, 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        .run_turn("read the requested files", &mut ui)
        .await
        .unwrap();
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.tool_calls, 2, "the denied suffix is not spent");
    assert_eq!(
        telemetry.tool_timeline.len(),
        3,
        "all model calls receive results"
    );
    assert_eq!(
        telemetry
            .tool_timeline
            .iter()
            .filter(|entry| entry.status == hi_tools::ToolStatus::Denied)
            .count(),
        1
    );
}
