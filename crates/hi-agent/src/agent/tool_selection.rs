//! Per-task and per-round tool advertisement.

use std::{collections::BTreeSet, sync::Arc};

use hi_ai::ToolSpec;

use crate::{
    AgentConfig, LspMode, TaskIntent, ToolSet, WriteSubagentPolicy,
    steering::{is_bounded_file_review, is_file_reference},
};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BackgroundToolAvailability {
    pub shell: bool,
    pub tasks: bool,
}

/// Build the tool set for a task. Dynamic selection deliberately fails open
/// for broad local questions: extra schema is cheap; losing workspace access is
/// not. Narrowly recognizable file tasks are the exception because advertising
/// unrelated search/planning/subagent schemas for them adds latency without
/// giving the model a useful next action.
pub(super) fn advertised_tools(
    config: &AgentConfig,
    task: Option<(&str, TaskIntent)>,
) -> Arc<[ToolSpec]> {
    // Callers without session state (construction and pure catalog tests) fail
    // open. Live agents use `advertised_tools_with_background` below so a
    // fresh session does not pay for polling schemas it cannot use.
    advertised_tools_with_background(
        config,
        task,
        BackgroundToolAvailability {
            shell: true,
            tasks: true,
        },
    )
}

pub(super) fn advertised_tools_with_background(
    config: &AgentConfig,
    task: Option<(&str, TaskIntent)>,
    background: BackgroundToolAvailability,
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
                // A read-only follow-up can poll a background process only
                // when this session still has a live handle for it. Never
                // advertise a polling schema for a fresh or completed handle:
                // models can otherwise invent stale ids and routed APIs may
                // reject the resulting call as an unknown tool.
                hi_tools::ToolCapability::Background => {
                    if mutating {
                        true
                    } else if !repo_relevant {
                        false
                    } else {
                        // A background schema is useful only while this
                        // session has a handle of the matching kind.
                        // Advertising `bash_output` on a fresh read-only turn
                        // makes models invent stale ids and adds avoidable
                        // schema tokens to every DeepSeek request.
                        match spec.name.as_str() {
                            "bash_output" => background.shell,
                            // The broad read-only catalog historically keeps
                            // kill out; the bounded direct-read path adds it
                            // only when an active shell is actually relevant.
                            "bash_kill" => false,
                            "get_task_output" | "wait_tasks" | "kill_task" => background.tasks,
                            _ => metadata.read_only,
                        }
                    }
                }
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
        // A bare repo-wide review is already a read-only, bounded inspection
        // task. DeepSeek tends to fan out three or four background reviews for
        // this shape and then block on `wait_tasks`, turning a useful answer
        // into a many-minute meta-loop. Keep subagents available when the user
        // explicitly requests parallel/delegated investigation, but make the
        // default review path inspect directly in the foreground.
        let suppress_new_subagents = task_text.is_some_and(|task| {
            broad_read_only_review(task, mutating) && !explicit_subagent_request(task)
        });
        // Explore: default-on for repo-relevant work; never for pure greetings.
        if !suppress_new_subagents
            && config.subagents.explore_subagents
            && (repo_relevant || matches!(config.memory.tool_set, ToolSet::Full))
        {
            specs.push(hi_tools::explore_tool_spec());
        }
        if !specs.is_empty() {
            specs.push(hi_tools::ask_user_tool_spec());
        }
        // Delegate: Off never; On for any mutation; Risk only isolation-shaped tasks.
        if !suppress_new_subagents && should_advertise_delegate(config, task_text, mutating) {
            specs.push(hi_tools::delegate_tool_spec());
        }
        // Background subagent tools: `task` spawns async subagents;
        // `get_task_output`/`wait_tasks`/`kill_task` poll/wait/cancel them.
        // Advertise when subagents are enabled and the task is repo-relevant.
        if config.subagents.explore_subagents
            && (repo_relevant || matches!(config.memory.tool_set, ToolSet::Full))
        {
            if !suppress_new_subagents {
                specs.push(hi_tools::task_tool_spec());
            }
            // Polling schemas are useful only after this session has actually
            // spawned a task. Advertising them on a fresh turn invites models
            // to invent task ids, and broad reviews can otherwise block on a
            // wait tool before doing any useful inspection.
            if !suppress_new_subagents || background.tasks {
                specs.push(hi_tools::get_task_output_tool_spec());
                specs.push(hi_tools::wait_tasks_tool_spec());
                specs.push(hi_tools::kill_task_tool_spec());
            }
        }
    }
    let direct_summary = task_text.is_some_and(|task| direct_file_summary_task(task, mutating));
    let direct_list = task_text.is_some_and(|task| direct_list_task(task, mutating));
    let direct_list_read =
        task_text.is_some_and(|task| direct_list_read_sequence_task(task, mutating));
    let bounded_review = task_text.is_some_and(|task| is_bounded_file_review(task, mutating));
    let targeted_mutation =
        task_text.is_some_and(|task| targeted_single_file_mutation_task(task, mutating));
    let targeted_multi_mutation =
        task_text.is_some_and(|task| targeted_multi_file_mutation_task(task, mutating));
    if targeted_mutation || targeted_multi_mutation {
        // A named small-scope edit needs evidence, an edit primitive, and a
        // way to run focused validation. It does not need repository census,
        // web/MCP/memory schemas, coordination, or subagent tools. Keep
        // multi_edit only for one-file work: its contract is atomic edits to
        // one file, and exposing it on a multi-file task invites the model to
        // pack unrelated paths into an invalid call before recovering with
        // separate edits. Planning and file creation are opt-in below: an
        // explicit update to several existing files still needs only the
        // evidence/edit/check path.
        let needs_search = task_text.is_some_and(targeted_mutation_needs_search);
        let needs_diff = task_text.is_some_and(targeted_mutation_needs_diff);
        let needs_plan = task_text.is_some_and(targeted_mutation_needs_plan);
        // A plan-driven recovery turn must retain write so the model can
        // transition from discovery to implementation after recording it.
        // Review/search-and-fix tasks may discover a replacement shape that
        // needs the full-file write primitive even when the named files
        // already exist. Plain update-only tasks remain on edit/apply_patch.
        let needs_write =
            needs_plan || needs_search || task_text.is_some_and(targeted_mutation_needs_write);
        let allows_shell = task_text.is_none_or(targeted_mutation_allows_shell);
        specs.retain(|spec| {
            matches!(spec.name.as_str(), "read" | "edit" | "apply_patch")
                || (needs_search && spec.name == "grep")
                || (needs_diff && spec.name == "diff")
                || (allows_shell && spec.name == "bash")
                || (needs_plan && spec.name == "update_plan")
                || (needs_write && spec.name == "write")
                || (targeted_mutation && spec.name == "multi_edit")
                || (background.shell && matches!(spec.name.as_str(), "bash_output" | "bash_kill"))
        });
    } else if direct_summary || direct_list || direct_list_read || bounded_review {
        // Keep background controls only when this session has actually
        // started that kind of work. A fresh direct-read request should send
        // just the tool it can use; an existing process/task retains its
        // non-mutating polling controls on a later read-only turn.
        specs.retain(|spec| {
            (spec.name == "read" && (direct_summary || direct_list_read || bounded_review))
                || (spec.name == "list" && (direct_list || direct_list_read))
                || (bounded_review && spec.name == "grep")
                || (background.shell && matches!(spec.name.as_str(), "bash_output" | "bash_kill"))
                || (background.tasks
                    && matches!(
                        spec.name.as_str(),
                        "get_task_output" | "wait_tasks" | "kill_task"
                    ))
        });
    }
    // Census-driven trim, applied last so it covers pushed-on tools too. The
    // protected floor is re-enforced here: the trim CLI already refuses floor
    // names, but a hand-edited or corrupted list must degrade to "no trim",
    // never to an agent that cannot read or edit.
    if !config.memory.disabled_tools.is_empty() {
        specs.retain(|spec| {
            hi_tools::PROTECTED_TOOLS.contains(&spec.name.as_str())
                || !config
                    .memory
                    .disabled_tools
                    .iter()
                    .any(|disabled| disabled == &spec.name)
        });
    }
    specs.into()
}

/// Whether the prompt explicitly requests one list operation and nothing that
/// needs a second repository tool. This is intentionally stricter than the
/// general dynamic catalog: a vague "list the issues" or a dependent
/// "list, then read" request must retain the broader catalog so the model can
/// continue without receiving an unknown-tool error.
fn direct_list_task(task: &str, mutating: bool) -> bool {
    if mutating {
        return false;
    }
    let lower = task.to_ascii_lowercase();
    let explicit_list = [
        "use the list tool",
        "call the list tool",
        "make one list call",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    if !explicit_list {
        return false;
    }
    if [
        "then read",
        "then grep",
        "then search",
        "after the list",
        "and read",
        "and grep",
        "and search",
        "multiple tool",
        "more than one tool",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        return false;
    }
    lower.contains("exactly one tool")
        || lower.contains("one tool call")
        || lower.contains("only one tool")
        || lower.contains("list only")
}

/// Whether the prompt explicitly requests the common repository orientation
/// sequence `list → read`. Keep this separate from the single-list fast path so
/// a dependent second call is never hidden from the model.
fn direct_list_read_sequence_task(task: &str, mutating: bool) -> bool {
    if mutating {
        return false;
    }
    let lower = task.to_ascii_lowercase();
    let has_list = lower.contains("use the list tool") || lower.contains("call the list tool");
    let has_read = lower.contains("use the read tool") || lower.contains("call the read tool");
    has_list
        && has_read
        && (lower.contains("then")
            || lower.contains("in that order")
            || lower.contains("both tool calls"))
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

/// Whether a prompt is a direct lookup whose only workspace action is reading
/// a small set of named files. Keep this intentionally conservative: reviews,
/// searches, dependency analysis, and explanations need the broader
/// search/orientation catalog.
fn direct_file_summary_task(task: &str, mutating: bool) -> bool {
    if mutating {
        return false;
    }
    let lower = task.to_ascii_lowercase();
    if [
        "review",
        "audit",
        "search",
        "grep",
        "where",
        "reference",
        "dependency",
        "related",
        "across",
        "directory",
        "folder",
        "project",
        "repository",
        "repo",
        "explain how",
        "how does",
        "why does",
        "issue",
        "bug",
        "list tool",
        "call list",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        return false;
    }
    let summary_request = [
        "read ",
        "read tool",
        "use the read tool",
        "call the read tool",
        "compare",
        "difference",
        "differences",
        "summarize",
        "summary",
        "purpose",
        "contents",
        "what is in",
        "what's in",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    if !summary_request {
        return false;
    }
    let file_mentions = lower
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-' | '/'))
        })
        .filter(|token| is_file_reference(token))
        .collect::<BTreeSet<_>>();
    (1..=4).contains(&file_mentions.len())
}

/// Count named files for a small-scope explicit mutation. Keep this narrower
/// than the general mutation intent: ambiguous prompts deliberately retain
/// the broad catalog, and multi-file/isolation-shaped work may need planning,
/// delegation, or repository orientation tools.
fn targeted_mutation_file_count(task: &str, mutating: bool) -> Option<usize> {
    if !mutating {
        return None;
    }
    let lower = task.to_ascii_lowercase();
    let explicit = crate::task_contract::explicit_mutation_request(&lower);
    // `delegate_risk_relevant` also treats any two source-file mentions as a
    // delegation candidate. That is useful for subagent admission, but too
    // conservative here: a direct request to change two named files can still
    // use a compact parent-run edit flow.
    let isolation = [
        "in parallel",
        "worktree",
        "isolated",
        "hand off",
        "handoff",
        "subagent",
        "delegate",
        "separately",
        "independent of",
        "multi-file",
        "multifile",
        "across crates",
        "across packages",
        "across modules",
        "whole crate",
        "entire package",
        "refactor",
        "migrate",
        "rewrite",
        "split into",
        "extract into",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
        || lower
            .split(|character: char| !character.is_ascii_alphanumeric())
            .any(|word| word == "port");
    let broad = [
        "across the repository",
        "across the codebase",
        "whole project",
        "entire project",
        "all files",
        "in parallel",
        "multiple files",
        "multi-file",
        "multifile",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    let file_mentions = lower
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-' | '/'))
        })
        .filter(|token| is_file_reference(token))
        .collect::<BTreeSet<_>>();
    (explicit && !isolation && !broad).then_some(file_mentions.len())
}

/// Whether a prompt is an explicit, single-file mutation.
fn targeted_single_file_mutation_task(task: &str, mutating: bool) -> bool {
    targeted_mutation_file_count(task, mutating) == Some(1)
}

/// Whether a prompt is an explicit, small multi-file mutation. Larger or
/// isolation-shaped changes retain the broad catalog so the model can plan,
/// search, or delegate when that is actually useful.
fn targeted_multi_file_mutation_task(task: &str, mutating: bool) -> bool {
    matches!(targeted_mutation_file_count(task, mutating), Some(2..=4))
}

/// Search is an optional branch for a known-file edit. Keep its schema out of
/// the common path unless the prompt explicitly asks for review/search work;
/// the model can inspect the named file directly with `read`.
fn targeted_mutation_needs_search(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    [
        "review", "audit", "grep", "search", "find", "locate", "look for", "symbol",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

/// A diff tool is useful when the user asks for a comparison, but it is
/// redundant for the usual edit flow: a final targeted `read` is cheaper and
/// gives the model the post-edit evidence it needs.
fn targeted_mutation_needs_diff(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    [
        "show the diff",
        "show diff",
        "git diff",
        "compare before and after",
        "compare the changes",
        "diff the changes",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

/// Planning is useful for an explicitly multi-step or long-horizon task, not
/// for a normal one-file edit. Keep the plan schema out of the common path so
/// DeepSeek does not spend tokens considering bookkeeping it was told to skip.
fn targeted_mutation_needs_plan(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    [
        "make a plan",
        "create a plan",
        "use a plan",
        "plan this",
        "plan",
        "checklist",
        "milestone",
        "several steps",
        "multiple steps",
        "multi-step",
        "break this down",
        "break it down",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

/// Keep the file-creation primitive for prompts that can reasonably require a
/// new file or a deliberate full overwrite. Existing-file updates use the
/// smaller edit primitives and can still fall back to the broad catalog when
/// the request is not a targeted mutation.
fn targeted_mutation_needs_write(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    [
        "create ",
        "create a new",
        "new file",
        "add a file",
        "write a file",
        "generate a file",
        "scaffold",
        "implement",
        "overwrite",
        "replace the entire file",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

/// Do not advertise a process tool when the user explicitly forbids shell
/// validation. Keeping the schema out of the request makes that constraint
/// enforceable instead of relying on the model to ignore an available tool.
fn targeted_mutation_allows_shell(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    ![
        "do not run shell",
        "don't run shell",
        "do not use shell",
        "don't use shell",
        "without running shell",
        "skip shell validation",
        "do not run validation",
        "don't run validation",
        "without running validation",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
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
    let has_port_word = lower
        .split(|character: char| !character.is_ascii_alphanumeric())
        .any(|word| word == "port");
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
        " port ",
        "rewrite",
        "split into",
        "extract into",
    ]
    .iter()
    .any(|m| lower.contains(m))
        || has_port_word
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

fn broad_read_only_review(task: &str, mutating: bool) -> bool {
    if mutating {
        return false;
    }
    let lower = task.to_ascii_lowercase();
    let review_verb = lower
        .split(|character: char| !character.is_ascii_alphanumeric())
        .next()
        .is_some_and(|word| matches!(word, "review" | "audit"));
    review_verb
        && [
            "codebase",
            "repository",
            "repo",
            "whole",
            "entire",
            "all files",
            "across",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
}

fn explicit_subagent_request(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    [
        "in parallel",
        "parallel investigation",
        "run in parallel",
        "parallel review",
        "subagent",
        "delegate",
        "explore subagent",
        "independent review",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
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
        assert!(
            names(&program).contains(&"bash_output"),
            "the catalog helper keeps background polling when availability is unknown: {:?}",
            names(&program)
        );
        assert!(!names(&program).contains(&"bash_kill"));
        let status = advertised_tools(&config, Some(("status", TaskIntent::ReadOnly)));
        assert!(
            names(&status).contains(&"bash_output"),
            "status follow-up tools: {:?}",
            names(&status)
        );
        assert!(!names(&status).contains(&"bash_kill"));
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
        assert!(
            names(&mutation).contains(&"ask_user"),
            "ask_user on interactive coding: {:?}",
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
    fn explicit_single_list_request_gets_only_the_list_schema() {
        let config = AgentConfig::default();
        let tools = advertised_tools_with_background(
            &config,
            Some((
                "Use the list tool on the repository root. Make exactly one tool call and report the first entry.",
                TaskIntent::ReadOnly,
            )),
            BackgroundToolAvailability::default(),
        );
        assert_eq!(names(&tools), vec!["list"]);

        let dependent = advertised_tools_with_background(
            &config,
            Some((
                "Use the list tool, then use the read tool on Cargo.toml, in that order.",
                TaskIntent::ReadOnly,
            )),
            BackgroundToolAvailability::default(),
        );
        assert_eq!(names(&dependent), vec!["read", "list"]);
    }

    #[test]
    fn disabled_tools_are_dropped_but_the_floor_survives_bad_lists() {
        let mut config = AgentConfig::default();
        config.memory.disabled_tools = vec![
            "glob".into(),
            "repo_map".into(),
            // A corrupted/hand-edited list naming core tools must be inert.
            "read".into(),
            "bash".into(),
        ];
        let tools = advertised_tools(
            &config,
            Some(("implement the parser", TaskIntent::Mutation)),
        );
        assert!(!names(&tools).contains(&"glob"));
        assert!(!names(&tools).contains(&"repo_map"));
        assert!(
            names(&tools).contains(&"read"),
            "floor: {:?}",
            names(&tools)
        );
        assert!(
            names(&tools).contains(&"bash"),
            "floor: {:?}",
            names(&tools)
        );
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
        assert!(!delegate_risk_relevant(
            "Update greeting.txt and report what changed."
        ));
    }

    #[test]
    fn direct_file_summaries_use_a_lean_read_catalog() {
        let config = AgentConfig::default();
        let tools = advertised_tools_with_background(
            &config,
            Some((
                "Read README.md and state its purpose in one concise sentence",
                TaskIntent::ReadOnly,
            )),
            BackgroundToolAvailability::default(),
        );
        let direct_names = names(&tools);
        assert_eq!(
            direct_names,
            vec!["read"],
            "fresh direct summary tools: {direct_names:?}"
        );
        assert!(
            !direct_names.contains(&"grep"),
            "direct summary tools: {direct_names:?}"
        );
        assert!(
            !direct_names.contains(&"explore"),
            "direct summary tools: {direct_names:?}"
        );
        assert!(
            !direct_names.contains(&"update_plan"),
            "direct summary tools: {direct_names:?}"
        );

        let explicit_tool_call = advertised_tools_with_background(
            &config,
            Some((
                "Use exactly one read tool call with the paths array for Cargo.toml and crates/hi-ai/Cargo.toml. Summarize both files.",
                TaskIntent::ReadOnly,
            )),
            BackgroundToolAvailability::default(),
        );
        let explicit_names = names(&explicit_tool_call);
        assert_eq!(explicit_names, vec!["read"]);
    }

    #[test]
    fn explicit_single_file_mutations_use_a_lean_edit_catalog() {
        let config = AgentConfig::default();
        let tools = advertised_tools_with_background(
            &config,
            Some((
                "Fix the bug in crates/hi-ai/src/openai/request.rs, then run the focused tests.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        assert_eq!(
            names(&tools),
            vec!["read", "edit", "multi_edit", "bash", "apply_patch"]
        );

        let reported = advertised_tools_with_background(
            &config,
            Some((
                "Update greeting.txt so it says exactly 'hello from DeepSeek'. Make the smallest safe edit and report what changed.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        assert_eq!(
            names(&reported),
            vec!["read", "edit", "multi_edit", "bash", "apply_patch"]
        );

        let no_shell = advertised_tools_with_background(
            &config,
            Some((
                "Update greeting.txt so it says exactly 'hello from DeepSeek'. Make the smallest safe edit, do not run shell validation yourself, and report what changed.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        assert_eq!(
            names(&no_shell),
            vec!["read", "edit", "multi_edit", "apply_patch"]
        );

        let multi_file = advertised_tools_with_background(
            &config,
            Some((
                "Update greeting.py so greeting() returns exactly 'Hi, {name}!', and update farewell.py so farewell() returns exactly 'See you, {name}!'. Make the smallest safe edits, run one relevant check, and report what changed.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        let multi_file_names = names(&multi_file);
        assert_eq!(
            multi_file_names,
            vec!["read", "edit", "bash", "apply_patch"]
        );
        assert!(
            !multi_file_names.contains(&"multi_edit"),
            "one-file atomic edit schema must not be advertised for multi-file work"
        );
        assert!(!multi_file_names.contains(&"write"));
        assert!(!multi_file_names.contains(&"update_plan"));

        let multi_create = advertised_tools_with_background(
            &config,
            Some((
                "Create new files greeting.py and farewell.py with small functions, then run a check.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        assert!(names(&multi_create).contains(&"write"));

        let multi_planned = advertised_tools_with_background(
            &config,
            Some((
                "Make a plan, then update greeting.py and farewell.py and run a check.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        assert!(names(&multi_planned).contains(&"update_plan"));
        assert!(names(&multi_planned).contains(&"write"));

        let create = advertised_tools_with_background(
            &config,
            Some((
                "Create a new file named greeting.txt with a short greeting.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        assert!(names(&create).contains(&"write"));

        let planned = advertised_tools_with_background(
            &config,
            Some((
                "Make a plan with several steps, then update greeting.txt and run a check.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        assert!(names(&planned).contains(&"update_plan"));

        let review = advertised_tools_with_background(
            &config,
            Some((
                "Review the bug in greeting.txt, search for the relevant symbol, then show the diff.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        let review_names = names(&review);
        assert!(
            review_names.contains(&"grep"),
            "review tools: {review_names:?}"
        );
        assert!(
            review_names.contains(&"diff"),
            "diff tools: {review_names:?}"
        );

        let broad = advertised_tools(
            &config,
            Some((
                "Refactor auth across src/a.rs and src/b.rs in parallel.",
                TaskIntent::Mutation,
            )),
        );
        assert!(
            names(&broad).contains(&"explore"),
            "broad mutation tools: {:?}",
            names(&broad)
        );
        assert!(
            names(&broad).contains(&"delegate"),
            "broad mutation tools: {:?}",
            names(&broad)
        );
    }

    #[test]
    fn file_reviews_keep_the_broad_inspection_catalog() {
        let config = AgentConfig::default();
        let tools = advertised_tools(
            &config,
            Some((
                "Review README.md and related code for accuracy issues",
                TaskIntent::ReadOnly,
            )),
        );
        let names = names(&tools);
        assert!(names.contains(&"read"), "review tools: {names:?}");
        assert!(names.contains(&"grep"), "review tools: {names:?}");
        assert!(names.contains(&"glob"), "review tools: {names:?}");
        assert!(names.contains(&"explore"), "review tools: {names:?}");
    }

    #[test]
    fn bounded_file_reviews_use_the_lean_read_search_catalog() {
        let config = AgentConfig::default();
        let tools = advertised_tools_with_background(
            &config,
            Some((
                "Review crates/hi-ai/src/openai/request.rs and crates/hi-ai/src/openai/stream.rs for one concrete bug. Use targeted read or grep within those two files only and give a fix recommendation.",
                TaskIntent::ReadOnly,
            )),
            BackgroundToolAvailability::default(),
        );
        assert_eq!(names(&tools), vec!["read", "grep"]);
    }

    #[test]
    fn bounded_review_does_not_narrow_broad_or_mutating_work() {
        let config = AgentConfig::default();
        let broad = advertised_tools(
            &config,
            Some((
                "Review the codebase for related issues across the repository.",
                TaskIntent::ReadOnly,
            )),
        );
        assert!(names(&broad).contains(&"grep"));

        let mutation = advertised_tools(
            &config,
            Some((
                "Review request.rs and fix the concrete bug in stream.rs.",
                TaskIntent::Mutation,
            )),
        );
        assert!(names(&mutation).contains(&"write"));
        assert!(names(&mutation).contains(&"grep"));
    }

    #[test]
    fn bare_broad_reviews_do_not_spawn_or_poll_background_subagents() {
        let config = AgentConfig::default();
        let tools = advertised_tools_with_background(
            &config,
            Some(("review codebase", TaskIntent::ReadOnly)),
            BackgroundToolAvailability::default(),
        );
        let review_names = names(&tools);
        assert!(
            review_names.contains(&"read"),
            "review tools: {review_names:?}"
        );
        assert!(
            review_names.contains(&"grep"),
            "review tools: {review_names:?}"
        );
        assert!(
            review_names.contains(&"repo_map"),
            "review tools: {review_names:?}"
        );
        assert!(
            !review_names.contains(&"explore"),
            "review tools: {review_names:?}"
        );
        assert!(
            !review_names.contains(&"task"),
            "review tools: {review_names:?}"
        );
        assert!(
            !review_names.contains(&"wait_tasks"),
            "review tools: {review_names:?}"
        );
        assert!(
            !review_names.contains(&"get_task_output"),
            "review tools: {review_names:?}"
        );
        assert!(
            !review_names.contains(&"kill_task"),
            "review tools: {review_names:?}"
        );

        let explicit = advertised_tools_with_background(
            &config,
            Some((
                "review codebase using parallel independent subagent investigations",
                TaskIntent::ReadOnly,
            )),
            BackgroundToolAvailability::default(),
        );
        let explicit_names = names(&explicit);
        assert!(explicit_names.contains(&"explore"));
        assert!(explicit_names.contains(&"task"));
        assert!(explicit_names.contains(&"wait_tasks"));
    }

    #[test]
    fn small_file_comparisons_use_the_lean_read_catalog() {
        let config = AgentConfig::default();
        let tools = advertised_tools_with_background(
            &config,
            Some((
                "Read README.md and Cargo.toml and compare their purpose",
                TaskIntent::ReadOnly,
            )),
            BackgroundToolAvailability {
                shell: true,
                tasks: true,
            },
        );
        let names = names(&tools);
        assert!(names.contains(&"read"), "comparison tools: {names:?}");
        assert!(
            names.contains(&"bash_output"),
            "background shell polling must remain available: {names:?}"
        );
        assert!(!names.contains(&"grep"), "comparison tools: {names:?}");
        assert!(!names.contains(&"explore"), "comparison tools: {names:?}");
        assert!(
            !names.contains(&"update_plan"),
            "comparison tools: {names:?}"
        );
    }

    #[test]
    fn direct_file_summaries_keep_existing_background_controls() {
        let config = AgentConfig::default();
        let tools = advertised_tools_with_background(
            &config,
            Some(("Read README.md and state its purpose", TaskIntent::ReadOnly)),
            BackgroundToolAvailability {
                shell: true,
                tasks: true,
            },
        );
        let names = names(&tools);
        assert!(names.contains(&"read"), "active summary tools: {names:?}");
        assert!(
            names.contains(&"bash_output"),
            "active summary tools: {names:?}"
        );
        assert!(
            names.contains(&"get_task_output"),
            "active summary tools: {names:?}"
        );
        assert!(
            names.contains(&"wait_tasks"),
            "active summary tools: {names:?}"
        );
        assert!(!names.contains(&"grep"), "active summary tools: {names:?}");
    }
}
