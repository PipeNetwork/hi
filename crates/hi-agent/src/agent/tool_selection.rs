//! Per-task and per-round tool advertisement.

use std::sync::Arc;

use hi_ai::ToolSpec;

use crate::{AgentConfig, LspMode, TaskIntent, ToolSet, WriteSubagentPolicy};

/// Build the tool set for a task. Dynamic selection deliberately fails open
/// for local questions: extra schema is cheap; losing workspace access is not.
pub(super) fn advertised_tools(
    config: &AgentConfig,
    task: Option<(&str, TaskIntent)>,
) -> Arc<[ToolSpec]> {
    if matches!(config.memory.tool_set, ToolSet::Minimal) {
        return hi_tools::MINIMAL_TOOL_SPECS.clone().into();
    }
    let (repo_relevant, web_relevant, mutating, task_text) =
        task.map_or((true, true, true, None), |(task, intent)| {
            let lower = task.to_ascii_lowercase();
            let mutating = intent == TaskIntent::Mutation;
            let web_relevant = web_relevant(&lower);
            let repo_relevant = repository_tools_relevant(task, intent);
            (repo_relevant, web_relevant, mutating, Some(task))
        });
    let mut specs = hi_tools::TOOL_SPECS
        .iter()
        .filter(|spec| {
            if matches!(config.memory.tool_set, ToolSet::Full) {
                return true;
            }
            let Some(metadata) = hi_tools::tool_metadata(&spec.name) else {
                return false;
            };
            match metadata.capability {
                hi_tools::ToolCapability::Coordination => {
                    mutating || (config.subagents.long_horizon && (repo_relevant || web_relevant))
                }
                hi_tools::ToolCapability::Repository => repo_relevant,
                hi_tools::ToolCapability::Mutation | hi_tools::ToolCapability::Process => mutating,
                hi_tools::ToolCapability::Background => mutating,
                hi_tools::ToolCapability::Lsp => {
                    repo_relevant && !matches!(config.gates.lsp_mode, LspMode::Off)
                }
                hi_tools::ToolCapability::Web => web_relevant && (mutating || metadata.read_only),
                hi_tools::ToolCapability::Subagent => false,
                hi_tools::ToolCapability::Mcp | hi_tools::ToolCapability::Memory => {
                    mutating || matches!(config.memory.tool_set, ToolSet::Full)
                }
                hi_tools::ToolCapability::Skill => {
                    repo_relevant || matches!(config.memory.tool_set, ToolSet::Full)
                }
            }
        })
        .cloned()
        .collect::<Vec<_>>();
    // `block_step` only means anything while a long-horizon goal is driving.
    // Advertising it on ordinary turns invites a model to declare hard work
    // "blocked" when there is no checklist to set the step aside on.
    if !config.subagents.long_horizon {
        specs.retain(|spec| spec.name != "block_step");
    }
    if !config.subagents.is_subagent {
        // Explore: default-on for repo-relevant work; never for pure greetings.
        if config.subagents.explore_subagents
            && (repo_relevant || matches!(config.memory.tool_set, ToolSet::Full))
        {
            specs.push(hi_tools::explore_tool_spec());
        }
        // Delegate: Off never; On for any mutation; Risk only isolation-shaped tasks.
        if should_advertise_delegate(config, task_text, mutating) {
            specs.push(hi_tools::delegate_tool_spec());
        }
        // Background subagent tools: `task` spawns async subagents;
        // `get_task_output`/`wait_tasks`/`kill_task` poll/wait/cancel them.
        // Advertise when subagents are enabled and the task is repo-relevant.
        if config.subagents.explore_subagents
            && (repo_relevant || matches!(config.memory.tool_set, ToolSet::Full))
        {
            specs.push(hi_tools::task_tool_spec());
            specs.push(hi_tools::get_task_output_tool_spec());
            specs.push(hi_tools::wait_tasks_tool_spec());
            specs.push(hi_tools::kill_task_tool_spec());
        }
    }
    specs.into()
}

fn should_advertise_delegate(config: &AgentConfig, task: Option<&str>, mutating: bool) -> bool {
    if matches!(config.memory.tool_set, ToolSet::Full) {
        return config.subagents.write_subagents.is_enabled();
    }
    match config.subagents.write_subagents {
        WriteSubagentPolicy::Off => false,
        WriteSubagentPolicy::On => mutating,
        // No task yet (startup refresh): fail open so the tool is present until
        // the first turn re-filters. With a task, only isolation-shaped work.
        WriteSubagentPolicy::Risk => match task {
            None => mutating,
            Some(task) => mutating && delegate_risk_relevant(task),
        },
    }
}

/// Heuristic: isolation pays for multi-file / multi-module / parallelizable work,
/// not for a one-line single-file fix the parent should do itself.
pub(super) fn delegate_risk_relevant(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    // Explicit isolation / parallel handoff language.
    if [
        "in parallel",
        "worktree",
        "isolated",
        "hand off",
        "handoff",
        "subagent",
        "delegate",
        "separately",
        "independent of",
    ]
    .iter()
    .any(|m| lower.contains(m))
    {
        return true;
    }
    // Multi-path or multi-crate shape in the prompt.
    let path_hits = lower
        .split_whitespace()
        .filter(|w| {
            w.contains('/')
                && (w.ends_with(".rs")
                    || w.ends_with(".py")
                    || w.ends_with(".ts")
                    || w.ends_with(".go")
                    || w.ends_with(".js")
                    || w.contains("src/")
                    || w.contains("crates/"))
        })
        .count();
    if path_hits >= 2 {
        return true;
    }
    // Multi-module / multi-package verbs.
    if [
        "multi-file",
        "multifile",
        "across crates",
        "across packages",
        "across modules",
        "whole crate",
        "entire package",
        "refactor",
        "migrate",
        "port ",
        "rewrite",
        "split into",
        "extract into",
    ]
    .iter()
    .any(|m| lower.contains(m))
    {
        return true;
    }
    // Several distinct source-file basename mentions (foo.rs + bar.rs).
    let file_names = lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.')
        .filter(|w| {
            w.ends_with(".rs")
                || w.ends_with(".py")
                || w.ends_with(".ts")
                || w.ends_with(".tsx")
                || w.ends_with(".go")
                || w.ends_with(".js")
                || w.ends_with(".jsx")
        })
        .collect::<std::collections::BTreeSet<_>>();
    file_names.len() >= 2
}

fn repository_tools_relevant(task: &str, intent: TaskIntent) -> bool {
    let lower = task.to_ascii_lowercase();
    intent == TaskIntent::Mutation
        || explicitly_repository_relevant(&lower)
        || (!externally_scoped(&lower) && !clearly_conversational(&lower))
}

fn externally_scoped(lower: &str) -> bool {
    lower.contains("http://")
        || lower.contains("https://")
        || lower
            .split(|character: char| !character.is_ascii_alphanumeric())
            .any(|word| matches!(word, "internet" | "online" | "web"))
}

fn web_relevant(lower: &str) -> bool {
    externally_scoped(lower)
        || ["latest", "current", "release notes", "documentation"]
            .iter()
            .any(|marker| lower.contains(marker))
}

fn explicitly_repository_relevant(lower: &str) -> bool {
    lower
        .split(|character: char| !character.is_ascii_alphanumeric())
        .any(|word| {
            matches!(
                word,
                "app"
                    | "add"
                    | "application"
                    | "audit"
                    | "binary"
                    | "build"
                    | "cargo"
                    | "class"
                    | "change"
                    | "code"
                    | "config"
                    | "crate"
                    | "create"
                    | "debug"
                    | "delete"
                    | "dependency"
                    | "edit"
                    | "file"
                    | "fix"
                    | "function"
                    | "implement"
                    | "manifest"
                    | "migrate"
                    | "module"
                    | "package"
                    | "program"
                    | "project"
                    | "refactor"
                    | "remove"
                    | "rename"
                    | "replace"
                    | "repo"
                    | "repository"
                    | "review"
                    | "source"
                    | "test"
                    | "update"
                    | "workspace"
                    | "write"
            )
        })
        || ["src/", ".go", ".js", ".py", ".rs", ".ts"]
            .iter()
            .any(|marker| lower.contains(marker))
}

fn clearly_conversational(lower: &str) -> bool {
    let normalized = lower
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    matches!(
        normalized.as_str(),
        "hi" | "hello"
            | "hey"
            | "thanks"
            | "thank you"
            | "good morning"
            | "good afternoon"
            | "good evening"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(tools: &Arc<[ToolSpec]>) -> Vec<&str> {
        tools.iter().map(|tool| tool.name.as_str()).collect()
    }

    #[test]
    fn dynamic_catalog_selects_task_relevant_capabilities() {
        let config = AgentConfig::default();
        let program = advertised_tools(
            &config,
            Some(("what does this program do", TaskIntent::ReadOnly)),
        );
        assert!(names(&program).contains(&"read"));
        assert!(names(&program).contains(&"list"));
        assert!(!names(&program).contains(&"write"));
        let web = advertised_tools(
            &config,
            Some(("fetch current documentation online", TaskIntent::ReadOnly)),
        );
        assert!(names(&web).contains(&"web_search"));
        assert!(!names(&web).contains(&"read"));
        assert!(!names(&web).contains(&"web_download"));

        let local_freshness = advertised_tools(
            &config,
            Some(("what changed in the latest commit", TaskIntent::ReadOnly)),
        );
        assert!(names(&local_freshness).contains(&"read"));
        assert!(names(&local_freshness).contains(&"web_search"));

        let mutation = advertised_tools(
            &config,
            Some(("implement the parser", TaskIntent::Mutation)),
        );
        assert!(names(&mutation).contains(&"write"));
        assert!(names(&mutation).contains(&"bash"));
        // Explore is default-on for repo-relevant coding.
        assert!(
            names(&mutation).contains(&"explore"),
            "explore on coding: {:?}",
            names(&mutation)
        );
        // Risk policy: single-file "implement the parser" is not isolation-shaped.
        assert!(
            !names(&mutation).contains(&"delegate"),
            "delegate not for simple mutation under risk: {:?}",
            names(&mutation)
        );
        let multi = advertised_tools(
            &config,
            Some((
                "refactor auth across src/a.rs and src/b.rs",
                TaskIntent::Mutation,
            )),
        );
        assert!(
            names(&multi).contains(&"delegate"),
            "delegate for multi-file risk: {:?}",
            names(&multi)
        );

        let mut long_horizon = config;
        long_horizon.subagents.long_horizon = true;
        let greeting = advertised_tools(&long_horizon, Some(("hello", TaskIntent::ReadOnly)));
        assert!(
            greeting.is_empty(),
            "greeting tools: {:?}",
            names(&greeting)
        );
        assert!(
            !names(&greeting).contains(&"explore"),
            "no explore on pure greeting"
        );
        for prompt in ["search the internet", "search the web"] {
            let tools = advertised_tools(&long_horizon, Some((prompt, TaskIntent::ReadOnly)));
            assert!(
                names(&tools).contains(&"web_search"),
                "{prompt}: {:?}",
                names(&tools)
            );
            assert!(
                !names(&tools).contains(&"read"),
                "{prompt}: {:?}",
                names(&tools)
            );
        }
    }

    #[test]
    fn delegate_risk_heuristic_matches_isolation_shape() {
        assert!(delegate_risk_relevant(
            "refactor auth across src/a.rs and src/b.rs"
        ));
        assert!(delegate_risk_relevant("migrate the crate to the new API"));
        assert!(delegate_risk_relevant("implement this in a worktree"));
        assert!(!delegate_risk_relevant("implement the parser"));
        assert!(!delegate_risk_relevant("fix the typo in main.rs"));
    }
}
