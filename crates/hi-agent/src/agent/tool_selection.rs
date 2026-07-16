//! Per-task and per-round tool advertisement.

use std::sync::Arc;

use hi_ai::ToolSpec;

use crate::{AgentConfig, LspMode, TaskIntent, ToolSet};

/// Build the tool set for a task. Dynamic selection deliberately fails open
/// for local questions: extra schema is cheap; losing workspace access is not.
pub(super) fn advertised_tools(
    config: &AgentConfig,
    task: Option<(&str, TaskIntent)>,
) -> Arc<[ToolSpec]> {
    if matches!(config.tool_set, ToolSet::Minimal) {
        return hi_tools::MINIMAL_TOOL_SPECS.clone().into();
    }
    let (repo_relevant, web_relevant, mutating) =
        task.map_or((true, true, true), |(task, intent)| {
            let lower = task.to_ascii_lowercase();
            let mutating = intent == TaskIntent::Mutation;
            let web_relevant = web_relevant(&lower);
            let repo_relevant = repository_tools_relevant(task, intent);
            (repo_relevant, web_relevant, mutating)
        });
    let mut specs = hi_tools::TOOL_SPECS
        .iter()
        .filter(|spec| {
            if matches!(config.tool_set, ToolSet::Full) {
                return true;
            }
            let Some(metadata) = hi_tools::tool_metadata(&spec.name) else {
                return false;
            };
            match metadata.capability {
                hi_tools::ToolCapability::Coordination => {
                    mutating || (config.long_horizon && (repo_relevant || web_relevant))
                }
                hi_tools::ToolCapability::Repository => repo_relevant,
                hi_tools::ToolCapability::Mutation | hi_tools::ToolCapability::Process => mutating,
                hi_tools::ToolCapability::Background => mutating,
                hi_tools::ToolCapability::Lsp => {
                    repo_relevant && !matches!(config.lsp_mode, LspMode::Off)
                }
                hi_tools::ToolCapability::Web => web_relevant && (mutating || metadata.read_only),
                hi_tools::ToolCapability::Subagent => false,
            }
        })
        .cloned()
        .collect::<Vec<_>>();
    if !config.is_subagent {
        if config.explore_subagents && (repo_relevant || matches!(config.tool_set, ToolSet::Full)) {
            specs.push(hi_tools::explore_tool_spec());
        }
        if config.write_subagents && (mutating || matches!(config.tool_set, ToolSet::Full)) {
            specs.push(hi_tools::delegate_tool_spec());
        }
    }
    specs.into()
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

        let mut long_horizon = config;
        long_horizon.long_horizon = true;
        let greeting = advertised_tools(&long_horizon, Some(("hello", TaskIntent::ReadOnly)));
        assert!(
            greeting.is_empty(),
            "greeting tools: {:?}",
            names(&greeting)
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
}
