//! Tests for the background subagent task system (`task`/`get_task_output`/
//! `wait_tasks`/`kill_task`).

use super::common::*;
use super::*;

fn bg_config() -> AgentConfig {
    let mut cfg = config();
    cfg.subagents.explore_subagents = true;
    cfg
}

struct BackgroundDropFlag(std::sync::Arc<std::sync::atomic::AtomicBool>);

struct BackgroundDelegateRunner;

#[async_trait::async_trait]
impl crate::DelegateRunner for BackgroundDelegateRunner {
    async fn run(&self, _task: &str, _verify: Option<&str>) -> crate::DelegateOutcome {
        unreachable!("background general-purpose children run in-process")
    }
}

impl Drop for BackgroundDropFlag {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[tokio::test]
async fn dropping_agent_aborts_its_active_background_tasks() {
    let agent = agent(Vec::new(), bg_config());
    // Retain the same kind of observer handle used by frontends while a turn
    // borrows the Agent. Agent drop must cancel its tasks even though this Arc
    // keeps the registry allocation alive.
    let registry = agent.background_task_registry();
    let started = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let started_in_task = started.clone();
    let dropped_in_task = dropped.clone();
    let task_id = registry
        .spawn(
            "agent-owned",
            "delegate",
            Box::new(move || {
                Box::pin(async move {
                    let _drop_flag = BackgroundDropFlag(dropped_in_task);
                    started_in_task.store(true, std::sync::atomic::Ordering::SeqCst);
                    std::future::pending::<()>().await;
                    unreachable!("agent-owned background task survived forever")
                })
            }),
        )
        .await
        .unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !started.load(std::sync::atomic::Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("background task should start");

    drop(agent);

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !dropped.load(std::sync::atomic::Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping Agent must drop its active background task future");

    let outcome = registry
        .poll(&task_id, std::time::Duration::ZERO)
        .await
        .expect("retained observer should still see the terminal task");
    assert_eq!(outcome.state, hi_tools::BackgroundTaskState::Cancelled);

    let spawn_after_owner_drop = registry
        .spawn(
            "orphan",
            "delegate",
            Box::new(|| Box::pin(std::future::pending::<hi_tools::BackgroundTaskOutcome>())),
        )
        .await;
    assert!(
        spawn_after_owner_drop.is_err(),
        "an observer handle must not restart work after Agent drop"
    );
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
async fn handle_task_unknown_subagent_type_fails() {
    let mut agent = agent(Vec::new(), bg_config());
    let mut ui = NullUi;
    let outcome = agent
        .handle_task(
            r#"{"description": "x", "prompt": "do something", "subagent_type": "wizard"}"#,
            &mut ui,
        )
        .await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
    assert!(
        outcome.content.contains("unknown subagent_type"),
        "got: {}",
        outcome.content
    );
}

#[tokio::test]
async fn handle_task_reports_registry_capacity_as_actionable_denial() {
    let mut agent = agent(Vec::new(), bg_config());
    let registry = agent.background_task_registry();
    for index in 0..16 {
        registry
            .spawn(
                &format!("capacity-{index}"),
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        std::future::pending::<()>().await;
                        unreachable!("capacity fixture must remain live")
                    })
                }),
            )
            .await
            .unwrap();
    }

    let mut ui = NullUi;
    let outcome = agent
        .handle_task(
            r#"{"description":"one more","prompt":"inspect one more thing","subagent_type":"explore"}"#,
            &mut ui,
        )
        .await;

    assert_eq!(outcome.status, hi_tools::ToolStatus::Denied);
    assert!(outcome.content.contains("background task capacity reached"));
    assert!(outcome.content.contains("get_task_output or wait_tasks"));
}

#[tokio::test]
async fn general_purpose_task_continues_an_incomplete_plan_after_a_recap() {
    let mut cfg = bg_config();
    cfg.memory.tool_set = ToolSet::Full;
    let root = cfg.paths.workspace_root.clone();
    let plan = |id: &str, first: &str, second: &str, third: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: serde_json::json!({
                    "steps": [
                        {"title": "write the first file", "status": first},
                        {"title": "write the second file", "status": second},
                        {"title": "write the third file", "status": third}
                    ]
                })
                .to_string(),
            }],
            1,
            1,
        )
    };
    let write = |id: &str, path: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "write".into(),
                arguments: serde_json::json!({"path": path, "content": path}).to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        plan("plan-start", "active", "pending", "pending"),
        write("write-first", "bg-first.txt"),
        plan("plan-middle", "done", "active", "pending"),
        completion(
            vec![Content::Text("The first step is complete.".into())],
            1,
            1,
        ),
        write("write-second", "bg-second.txt"),
        plan("plan-late", "done", "done", "active"),
        completion(
            vec![Content::Text("The second step is complete.".into())],
            1,
            1,
        ),
        write("write-third", "bg-third.txt"),
        plan("plan-done", "done", "done", "done"),
        completion(
            vec![Content::Text("All background steps are complete.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    agent.set_delegate_runner(std::sync::Arc::new(BackgroundDelegateRunner));
    let mut ui = NullUi;
    let spawned = agent
        .handle_task(
            r#"{"description":"three writes","prompt":"Implement all three file writes as a multi-step plan.","subagent_type":"general-purpose"}"#,
            &mut ui,
        )
        .await;
    assert_eq!(spawned.status, hi_tools::ToolStatus::Succeeded);
    let task_id = spawned
        .content
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("general-purpose task spawned: "))
        .expect("spawned task id");
    let result = agent
        .handle_wait_tasks(
            &serde_json::json!({
                "task_ids": [task_id],
                "timeout_ms": 5_000
            })
            .to_string(),
        )
        .await;

    assert!(
        result.content.contains("— Completed:"),
        "child stopped before its second productive step: {}",
        result.content
    );
    assert!(root.join("bg-first.txt").is_file());
    assert!(root.join("bg-second.txt").is_file());
    assert!(root.join("bg-third.txt").is_file());
}

#[test]
fn task_tool_spec_lists_grok_build_kinds() {
    let spec = hi_tools::task_tool_spec();
    let enum_vals = spec
        .parameters
        .pointer("/properties/subagent_type/enum")
        .and_then(|v| v.as_array())
        .expect("subagent_type enum");
    let names: Vec<&str> = enum_vals.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(names, vec!["explore", "plan", "general-purpose"]);
    assert!(spec.description.contains("explore"));
    assert!(spec.description.contains("plan"));
    assert!(spec.description.contains("general-purpose"));
    // scope was advertised but never enforced — removed until BG parallel
    // admission exists. Live-tree GP semantics are documented instead.
    assert!(spec.parameters.pointer("/properties/scope").is_none());
    assert!(
        spec.description.contains("live working tree") || spec.description.contains("live tree"),
        "GP isolation caveat missing from task tool description"
    );
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
    assert!(hi_tools::is_known_tool("memory_update"));
    assert!(hi_tools::is_known_tool("memory_forget"));
    assert!(hi_tools::is_known_tool("skill"));
}

#[test]
fn mcp_memory_skill_tool_specs_exist() {
    assert_eq!(hi_tools::use_tool_tool_spec().name, "use_tool");
    assert_eq!(hi_tools::search_tool_tool_spec().name, "search_tool");
    assert_eq!(hi_tools::memory_search_tool_spec().name, "memory_search");
    assert_eq!(hi_tools::memory_get_tool_spec().name, "memory_get");
    assert_eq!(hi_tools::memory_update_tool_spec().name, "memory_update");
    assert_eq!(hi_tools::memory_forget_tool_spec().name, "memory_forget");
    assert_eq!(hi_tools::skill_tool_spec().name, "skill");
}
