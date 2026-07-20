//! Tests for the read-only `explore` subagent ([`Agent::handle_explore`]).

use super::common::*;
use super::*;

fn explore_config() -> AgentConfig {
let mut cfg = config();
    cfg.subagents.explore_subagents = true;
    cfg
}

#[test]
fn explore_tool_is_not_read_only_and_not_in_global_set() {
    // Depth ≤ 1 is structural: `explore` is excluded from the read-only set (so a
    // read-only child never sees it) and from the global set (so it's only ever
    // advertised when explicitly injected for a capable parent).
    assert!(!hi_tools::is_read_only("explore"));
    assert!(!hi_tools::TOOL_SPECS.iter().any(|t| t.name == "explore"));
    // ...and it exists as an injectable spec.
    assert_eq!(hi_tools::explore_tool_spec().name, "explore");
}

#[test]
fn read_only_parent_keeps_explore() {
    // A top-level agent (is_subagent = false) keeps `explore` even in a read-only
    // / review turn — delegating a read-only investigation is itself read-only.
    let agent = agent(Vec::new(), explore_config());
    let auto = agent.request_tools_for(hi_ai::ToolMode::Auto);
    assert!(
        auto.iter().any(|t| t.name == "explore"),
        "auto advertises explore"
    );
    let read_only = agent.request_tools_for(hi_ai::ToolMode::ReadOnly);
    assert!(
        read_only.iter().any(|t| t.name == "explore"),
        "a read-only top-level turn must still offer explore"
    );
}

#[test]
fn subagent_never_gets_explore() {
    // A subagent (is_subagent = true) is never advertised `explore`, in any mode —
    // so it cannot spawn another (depth ≤ 1).
    let mut cfg = explore_config();
    cfg.subagents.is_subagent = true;
    let agent = agent(Vec::new(), cfg);
    for mode in [hi_ai::ToolMode::Auto, hi_ai::ToolMode::ReadOnly] {
        assert!(
            !agent
                .request_tools_for(mode)
                .iter()
                .any(|t| t.name == "explore"),
            "a subagent must never be offered explore ({mode:?})"
        );
    }
}

#[tokio::test]
async fn explore_missing_task_errors() {
    let mut agent = agent(Vec::new(), explore_config());
    let mut ui = NullUi;
    let out = agent.handle_explore("{}", &mut ui).await;
    assert_eq!(out.status, hi_tools::ToolStatus::Failed);
    assert!(out.content.contains("missing"), "got: {}", out.content);
    assert_eq!(agent.explore_subagents_used, 0);
}

#[tokio::test]
async fn explore_respects_session_budget() {
    let mut agent = agent(Vec::new(), explore_config());
    agent.explore_subagents_used = crate::agent::MAX_EXPLORE_SUBAGENTS_PER_SESSION;
    let mut ui = NullUi;
    let out = agent
        .handle_explore(r#"{"task":"anything"}"#, &mut ui)
        .await;
    assert_eq!(out.status, hi_tools::ToolStatus::Denied);
    assert!(
        out.content.contains("budget exhausted"),
        "got: {}",
        out.content
    );
    // Cap is not exceeded (no model call was made).
    assert_eq!(
        agent.explore_subagents_used,
        crate::agent::MAX_EXPLORE_SUBAGENTS_PER_SESSION
    );
}

#[tokio::test]
async fn explore_runs_child_and_returns_answer() {
    // Parent and child share the same canned provider (Arc), popping in exactly
    // this order: [0] parent calls explore, [1] child's answer, [2] parent's final
    // reply. An exact count means a regression that adds a model call panics here.
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "e1".into(),
                name: "explore".into(),
                arguments: r#"{"task":"where is X configured?"}"#.into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "X is configured in config.toml under the [x] section.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text("Done — X lives in config.toml.".into())],
            1,
            1,
        ),
    ];
    let base = std::env::temp_dir().join(format!(
        "hi-explore-outcome-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let root = base.join("workspace");
    std::fs::create_dir_all(&root).unwrap();
    let mut cfg = explore_config();
    cfg.paths.workspace_root = root;
    cfg.paths.state_root = base.join("state");
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        .run_turn("find where X is configured", &mut ui)
        .await
        .unwrap();

    // Exactly one subagent ran, and its answer came back as the `explore` tool result.
    assert_eq!(agent.explore_subagents_used, 1);
    assert!(
        ui.tool_results
            .iter()
            .any(|(name, result)| name == "explore"
                && result.contains("config.toml under the [x] section")),
        "the child's answer should be returned as the explore tool result; got {:?}",
        ui.tool_results
    );
    // The prominent subagent callout fired (subagent_note falls back to status,
    // which RecUi records).
    assert!(
        ui.statuses.iter().any(|s| s.contains("explore subagent")),
        "expected a subagent callout in statuses; got {:?}",
        ui.statuses
    );
    let entry = agent
        .last_turn_telemetry()
        .tool_timeline
        .iter()
        .find(|entry| entry.tool == "explore")
        .expect("explore appears in the typed tool timeline");
    assert_eq!(entry.status, hi_tools::ToolStatus::Succeeded);
    assert!(entry.process.is_none());
    assert!(!entry.effects.mutation_attempted);
    assert_eq!(entry.truncation, hi_tools::TruncationState::Complete);
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn explore_mutation_wording_keeps_reads_read_only_and_succeeds() {
    let workspace = IsolatedWorkspace::new("explore-build-next-read-only");
    let mut responses = Vec::new();
    for batch in 0..7 {
        let mut calls = Vec::new();
        for index in 0..2 {
            let file = batch * 2 + index;
            let relative = format!("source-{file}.rs");
            std::fs::write(workspace.path(&relative), format!("evidence {file}\n")).unwrap();
            calls.push(Content::ToolCall {
                id: format!("read-{file}"),
                name: "read".into(),
                arguments: serde_json::json!({"path": relative}).to_string(),
            });
        }
        responses.push(completion(calls, 1, 1));
    }
    responses.push(completion(
        vec![Content::Text(
            "The next component to build is supported by source-13.rs.".into(),
        )],
        1,
        1,
    ));

    let mut cfg = workspace.config();
    cfg.subagents.explore_subagents = true;
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    let outcome = agent
        .handle_explore(
            r#"{"task":"review the current architecture and identify what to build next"}"#,
            &mut ui,
        )
        .await;

    assert_eq!(outcome.status, hi_tools::ToolStatus::Succeeded);
    assert!(outcome.content.contains("source-13.rs"));
    let reads = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "explore:read")
        .collect::<Vec<_>>();
    assert_eq!(reads.len(), 14, "all read-only investigation must run");
    assert!(
        reads
            .iter()
            .all(|(_, result)| !result.to_ascii_lowercase().contains("denied")),
        "read-only exploration must not manufacture denials: {reads:?}"
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("incomplete")
                || status.contains("forcing an edit")
                || status.contains("mutation request")),
        "mutation steering leaked into an explore child: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn explore_batched_failed_offset_reads_are_bounded_before_chat_only_answer() {
    let workspace = IsolatedWorkspace::new("explore-batched-sprawl");
    let paths = (0..30)
        .map(|index| {
            let path = workspace.path(format!("source-{index}.rs"));
            std::fs::write(&path, format!("file {index} contents\n")).unwrap();
            path
        })
        .collect::<Vec<_>>();
    let read_batch = |batch: usize, paths: &[std::path::PathBuf]| {
        completion(
            paths
                .iter()
                .enumerate()
                .map(|(index, path)| Content::ToolCall {
                    id: format!("read-{batch}-{index}"),
                    name: "read".into(),
                    // This mirrors the incident: the explorer kept probing an
                    // offset past EOF. Failed probes must still consume its
                    // inspection budget.
                    arguments: serde_json::json!({ "path": path, "offset": 2001 }).to_string(),
                })
                .collect(),
            1,
            1,
        )
    };
    let responses = vec![
        read_batch(0, &paths[0..10]),
        read_batch(1, &paths[10..20]),
        // This batch is proposed after the bounded Review inspection cap has
        // been crossed. It must be suppressed, not executed.
        read_batch(2, &paths[20..30]),
        completion(
            vec![Content::Text(
                "The bounded investigation found the relevant evidence in source-0.rs.".into(),
            )],
            1,
            1,
        ),
    ];
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut cfg = workspace.config();
    cfg.subagents.explore_subagents = true;
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    let outcome = agent
        .handle_explore(r#"{"task":"summarize the relevant source files"}"#, &mut ui)
        .await;

    assert_eq!(outcome.status, hi_tools::ToolStatus::Succeeded);
    assert!(outcome.content.contains("bounded investigation"));
    let read_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "explore:read")
        .collect::<Vec<_>>();
    assert_eq!(
        read_results.len(),
        20,
        "the post-cap batch must not execute: {:?}",
        ui.tool_results
    );
    assert!(
        read_results
            .iter()
            .all(|(_, result)| result.contains("past the end")),
        "fixture must exercise failed offset probes: {read_results:?}"
    );
    assert!(
        read_results
            .iter()
            .all(|(_, result)| !result.contains("file 20 contents")),
        "the first post-cap file was unexpectedly read: {read_results:?}"
    );
    assert_eq!(
        modes.lock().unwrap().as_slice(),
        [
            ToolMode::ReadOnly,
            ToolMode::ReadOnly,
            ToolMode::ReadOnly,
            ToolMode::ChatOnly,
        ],
        "the synthesis round must be forced chat-only"
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("reached step limit")),
        "the child should synthesize before its step cap: {:?}",
        ui.statuses
    );
}
