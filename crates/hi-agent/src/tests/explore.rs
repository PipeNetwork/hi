//! Tests for the read-only `explore` subagent ([`Agent::handle_explore`]).

use super::common::*;
use super::*;

fn explore_config() -> AgentConfig {
    let mut cfg = config();
    cfg.explore_subagents = true;
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
    cfg.is_subagent = true;
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
    assert!(out.contains("missing"), "got: {out}");
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
    assert!(out.contains("budget exhausted"), "got: {out}");
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
    let mut agent = agent(responses, explore_config());
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
}
