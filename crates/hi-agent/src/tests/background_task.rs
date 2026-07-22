//! Tests for the background subagent task system (`task`/`get_task_output`/
//! `wait_tasks`/`kill_task`).

use super::common::*;
use super::*;

fn bg_config() -> AgentConfig {
    let mut cfg = config();
    cfg.subagents.explore_subagents = true;
    cfg
}

#[test]
fn task_tool_spec_exists_and_is_not_in_global_set() {
    assert!(!hi_tools::TOOL_SPECS.iter().any(|t| t.name == "task"));
    assert_eq!(hi_tools::task_tool_spec().name, "task");
    assert_eq!(
        hi_tools::get_task_output_tool_spec().name,
        "get_task_output"
    );
    assert_eq!(hi_tools::wait_tasks_tool_spec().name, "wait_tasks");
    assert_eq!(hi_tools::kill_task_tool_spec().name, "kill_task");
}

#[test]
fn task_tools_are_in_catalog() {
    assert!(hi_tools::is_known_tool("task"));
    assert!(hi_tools::is_known_tool("get_task_output"));
    assert!(hi_tools::is_known_tool("wait_tasks"));
    assert!(hi_tools::is_known_tool("kill_task"));
}

#[test]
fn task_tools_advertised_for_top_level_agent() {
    let agent = agent(Vec::new(), bg_config());
    let tools = agent.request_tools_for(hi_ai::ToolMode::Auto);
    assert!(
        tools.iter().any(|t| t.name == "task"),
        "task tool should be advertised for a top-level agent"
    );
    assert!(
        tools.iter().any(|t| t.name == "get_task_output"),
        "get_task_output should be advertised"
    );
    assert!(
        tools.iter().any(|t| t.name == "wait_tasks"),
        "wait_tasks should be advertised"
    );
    assert!(
        tools.iter().any(|t| t.name == "kill_task"),
        "kill_task should be advertised"
    );
}

#[test]
fn subagent_never_gets_task_tools() {
    let mut cfg = bg_config();
    cfg.subagents.is_subagent = true;
    let agent = agent(Vec::new(), cfg);
    for mode in [hi_ai::ToolMode::Auto, hi_ai::ToolMode::ReadOnly] {
        assert!(
            !agent
                .request_tools_for(mode)
                .iter()
                .any(|t| t.name == "task"),
            "a subagent must never see the task tool (depth cap)"
        );
    }
}

#[tokio::test]
async fn handle_task_missing_prompt_fails() {
    let mut agent = agent(Vec::new(), bg_config());
    let mut ui = NullUi;
    let outcome = agent
        .handle_task(r#"{"description": "test", "prompt": ""}"#, &mut ui)
        .await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
}

#[tokio::test]
async fn handle_task_missing_description_fails() {
    let mut agent = agent(Vec::new(), bg_config());
    let mut ui = NullUi;
    let outcome = agent
        .handle_task(r#"{"description": "", "prompt": "do something"}"#, &mut ui)
        .await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
}

#[tokio::test]
async fn handle_kill_task_unknown_id_fails() {
    let agent = agent(Vec::new(), bg_config());
    let outcome = agent
        .handle_kill_task(r#"{"task_id": "nonexistent"}"#)
        .await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
}

#[tokio::test]
async fn handle_get_task_output_invalid_json_fails() {
    let agent = agent(Vec::new(), bg_config());
    let outcome = agent.handle_get_task_output("not json").await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
}

#[tokio::test]
async fn handle_wait_tasks_empty_ids_fails() {
    let agent = agent(Vec::new(), bg_config());
    let outcome = agent.handle_wait_tasks(r#"{"task_ids": []}"#).await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
}

#[test]
fn mcp_memory_skill_tools_in_catalog() {
    assert!(hi_tools::is_known_tool("use_tool"));
    assert!(hi_tools::is_known_tool("search_tool"));
    assert!(hi_tools::is_known_tool("memory_search"));
    assert!(hi_tools::is_known_tool("memory_get"));
    assert!(hi_tools::is_known_tool("skill"));
}

#[test]
fn mcp_memory_skill_tool_specs_exist() {
    assert_eq!(hi_tools::use_tool_tool_spec().name, "use_tool");
    assert_eq!(hi_tools::search_tool_tool_spec().name, "search_tool");
    assert_eq!(hi_tools::memory_search_tool_spec().name, "memory_search");
    assert_eq!(hi_tools::memory_get_tool_spec().name, "memory_get");
    assert_eq!(hi_tools::skill_tool_spec().name, "skill");
}
