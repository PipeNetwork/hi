//! Per-task and per-round tool advertisement.

mod task_shape;

pub(super) use task_shape::targeted_named_file_mutation;
use task_shape::*;

use std::sync::Arc;

use hi_ai::ToolSpec;

use crate::{AgentConfig, LspMode, TaskIntent, ToolSet, steering::is_bounded_file_review};

#[derive(Clone, Copy, Debug)]
pub(super) struct BackgroundToolAvailability {
    pub shell: bool,
    pub tasks: bool,
    /// Interactive TTY (TUI/REPL). Headless one-shot, `--loops-daemon`, and
    /// `hi mcp serve` leave this false so `browser_exec` is not advertised.
    pub interactive: bool,
}

impl Default for BackgroundToolAvailability {
    fn default() -> Self {
        Self {
            shell: false,
            tasks: false,
            interactive: true,
        }
    }
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
            interactive: true,
        },
    )
}

pub(super) fn advertised_tools_with_background(
    config: &AgentConfig,
    task: Option<(&str, TaskIntent)>,
    background: BackgroundToolAvailability,
) -> Arc<[ToolSpec]> {
    // Explicitly tool-free answer contracts are the one task shape where
    // failing closed is correct for every catalog mode, including
    // Full/Minimal. Advertising even a read tool invites repository wandering
    // and violates the user's response contract.
    if task.is_some_and(|(text, intent)| {
        intent == TaskIntent::ReadOnly
            && crate::task_contract::prompt_requests_tool_free_response(text)
    }) {
        return Vec::<ToolSpec>::new().into();
    }
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
                hi_tools::ToolCapability::Structure => true,
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
        if suppress_new_subagents {
            specs.retain(|spec| {
                !matches!(
                    spec.name.as_str(),
                    "update_plan" | "record_decision" | "block_step"
                )
            });
        }
        // Explore: default-on for repo-relevant work; never for pure greetings.
        if !suppress_new_subagents
            && config.subagents.explore_subagents
            && (repo_relevant || matches!(config.memory.tool_set, ToolSet::Full))
        {
            specs.push(hi_tools::explore_tool_spec());
        }
        if config.memory.offer_ask_user && !specs.is_empty() {
            specs.push(hi_tools::ask_user_tool_spec());
        }
        // Two gateway schemas instead of flattening every MCP tool onto the
        // request. Isolation/targeted retain below still drops them.
        if config.memory.offer_mcp && !specs.is_empty() {
            specs.push(hi_tools::search_tool_tool_spec());
            specs.push(hi_tools::use_tool_tool_spec());
        }
        // Inject-gated like MCP: isolation retain below drops these on
        // named-file edits. Parent sessions only — subagents must not rewrite
        // the user's markdown memory.
        if config.memory.offer_memory && !specs.is_empty() {
            specs.push(hi_tools::memory_search_tool_spec());
            specs.push(hi_tools::memory_get_tool_spec());
            specs.push(hi_tools::memory_update_tool_spec());
            specs.push(hi_tools::memory_forget_tool_spec());
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
    match research_injection(config, task_text) {
        ResearchInjection::None => {}
        ResearchInjection::ReadOnly => {
            specs.push(hi_tools::research_read_tool_spec());
        }
        ResearchInjection::SearchAndRead => {
            specs.push(hi_tools::research_tool_spec());
            specs.push(hi_tools::research_read_tool_spec());
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
    // `targeted_named_file_mutation` is the union of the two predicates above;
    // keep the split so one-file work can still advertise `multi_edit`.
    if targeted_mutation || targeted_multi_mutation {
        // A named small-scope edit needs evidence, an edit primitive, and a
        // way to run focused validation. It does not need repository census,
        // MCP/memory schemas, coordination, or subagent tools. Web fetch/search
        // stay only when the prompt itself is a URL or research write. Keep
        // multi_edit only for one-file work: its contract is atomic edits to
        // one file, and exposing it on a multi-file task invites the model to
        // pack unrelated paths into an invalid call before recovering with
        // separate edits. Planning and file creation are opt-in below: an
        // explicit update to several existing files still needs only the
        // evidence/edit/check path.
        let needs_search = task_text.is_some_and(targeted_mutation_needs_search);
        let needs_diff = task_text.is_some_and(targeted_mutation_needs_diff);
        let needs_plan = task_text.is_some_and(targeted_mutation_needs_plan);
        let needs_web_fetch = task_text.is_some_and(targeted_mutation_needs_web_fetch);
        let needs_web_search = task_text.is_some_and(targeted_mutation_needs_web_search);
        let research_read_only = matches!(
            research_injection(config, task_text),
            ResearchInjection::ReadOnly
        );
        // A plan-driven recovery turn must retain write so the model can
        // transition from discovery to implementation after recording it.
        // Review/search-and-fix tasks may discover a replacement shape that
        // needs the full-file write primitive even when the named files
        // already exist. Plain update-only tasks remain on edit/apply_patch.
        let needs_write =
            needs_plan || needs_search || task_text.is_some_and(targeted_mutation_needs_write);
        // A known public URL should go through `web_fetch`, not `bash` curl.
        let allows_shell = !needs_web_fetch && task_text.is_none_or(targeted_mutation_allows_shell);
        specs.retain(|spec| {
            matches!(spec.name.as_str(), "read" | "edit" | "apply_patch")
                || (needs_search && spec.name == "grep")
                || (needs_diff && spec.name == "diff")
                || (allows_shell && spec.name == "bash")
                || (needs_plan && spec.name == "update_plan")
                || (needs_write && spec.name == "write")
                || (needs_web_fetch && spec.name == "web_fetch")
                || (needs_web_search && spec.name == "web_search")
                || (needs_web_search && !research_read_only && spec.name == "research")
                || ((needs_web_search || research_read_only) && spec.name == "research_read")
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
    // Inject-gated: page/login/UI tasks only, and only when `[browser] enabled`.
    // Pushed after isolation retain so a named-file CSS/login edit still gets
    // the schema; census trim below can still drop it.
    if config.memory.offer_browser && background.interactive && should_advertise_browser(task_text)
    {
        specs.push(hi_tools::browser_exec_tool_spec());
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
    slim_pipe_flash_dynamic_tools(config, &mut specs);
    specs.into()
}

fn is_pipe_deepseek_v4_flash_model(model: &str) -> bool {
    matches!(
        model.trim(),
        "pipe/deepseek-v4-flash-vision-exp"
            | "pipe/deepseek-v4-flash-0731"
            | "pipe/deepseek-v4-flash"
    )
}

fn slim_pipe_flash_dynamic_tools(config: &AgentConfig, specs: &mut Vec<ToolSpec>) {
    if !matches!(config.memory.tool_set, ToolSet::Dynamic) {
        return;
    }
    if !is_pipe_deepseek_v4_flash_model(&config.routing.model) {
        return;
    }
    specs.retain(|spec| {
        matches!(
            spec.name.as_str(),
            "read"
                | "write"
                | "edit"
                | "multi_edit"
                | "apply_patch"
                | "run_program"
                | "bash"
                | "bash_output"
                | "bash_kill"
                | "list"
                | "grep"
                | "glob"
                | "repo_map"
                | "update_plan"
                | "record_decision"
                | "block_step"
                | "web_search"
                | "web_fetch"
                | "web_download"
                | "research"
                | "research_read"
                | "browser_exec"
                | "ask_user"
                | "new_context"
                | "diagnostics"
                | "definition"
                | "references"
                | "hover"
        )
    });
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

        let known_url = advertised_tools(
            &config,
            Some((
                "Fetch https://example.com/ at that exact URL",
                TaskIntent::ReadOnly,
            )),
        );
        assert!(
            names(&known_url).contains(&"web_fetch"),
            "a concrete public URL should advertise web_fetch: {:?}",
            names(&known_url)
        );
        assert!(
            names(&known_url).contains(&"web_search"),
            "web_search stays available when a URL is present: {:?}",
            names(&known_url)
        );

        let known_url_write = advertised_tools_with_background(
            &config,
            Some((
                "Fetch https://example.com/ at that exact URL. Do not search the web \
                 and do not ask the user. Write the page's main heading to answer.txt \
                 as the only line.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        let known_url_write_names = names(&known_url_write);
        assert!(
            known_url_write_names.contains(&"web_fetch"),
            "write-after-fetch must still advertise web_fetch: {known_url_write_names:?}"
        );
        assert!(
            known_url_write_names.contains(&"write"),
            "write-after-fetch must advertise write: {known_url_write_names:?}"
        );
        assert!(
            !known_url_write_names.contains(&"bash"),
            "write-after-fetch must not offer bash curl: {known_url_write_names:?}"
        );
        assert!(
            !known_url_write_names.contains(&"web_search"),
            "an exact URL should not open web_search: {known_url_write_names:?}"
        );

        let web_research_write = advertised_tools_with_background(
            &config,
            Some((
                "Research current Zig 0.16 HTTP client behavior on the web. Do not \
                 inspect this workspace and do not fetch a guessed URL first. Write \
                 one cited https source URL to answer.txt as the only line.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        let web_research_names = names(&web_research_write);
        assert!(
            web_research_names.contains(&"web_search"),
            "web research write must advertise web_search: {web_research_names:?}"
        );
        if hi_tools::research_credentials_configured() {
            assert!(
                web_research_names.contains(&"research")
                    && web_research_names.contains(&"research_read"),
                "Pipe research tools when credentials exist: {web_research_names:?}"
            );
        } else {
            assert!(
                !web_research_names.contains(&"research"),
                "research stays off without Pipe credentials: {web_research_names:?}"
            );
        }
        assert!(
            !web_research_names.contains(&"web_fetch"),
            "web research write must not guess-fetch: {web_research_names:?}"
        );
        let rsi = advertised_tools_with_background(
            &{
                let mut cfg = AgentConfig::default();
                cfg.rsi.managed = true;
                cfg
            },
            Some((
                "Research current Zig 0.16 HTTP client behavior on the web. Write \
                 one cited https source URL to answer.txt as the only line.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        assert!(
            !names(&rsi).contains(&"research"),
            "RSI managed code.change must not inject research: {:?}",
            names(&rsi)
        );

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
        assert!(
            !names(&mutation).contains(&"research"),
            "ordinary code.change must not inject research: {:?}",
            names(&mutation)
        );
        // Explore is default-on for repo-relevant coding.
        assert!(
            names(&mutation).contains(&"explore"),
            "explore on coding: {:?}",
            names(&mutation)
        );
        assert!(
            !names(&mutation).contains(&"ask_user"),
            "ask_user stays off for autonomous coding: {:?}",
            names(&mutation)
        );
        let mut interactive_config = AgentConfig::default();
        interactive_config.memory.offer_ask_user = true;
        let interactive_mutation = advertised_tools(
            &interactive_config,
            Some(("implement the parser", TaskIntent::Mutation)),
        );
        assert!(
            names(&interactive_mutation).contains(&"ask_user"),
            "ask_user is available only after explicit opt-in: {:?}",
            names(&interactive_mutation)
        );
        assert!(
            !names(&mutation).contains(&"search_tool") && !names(&mutation).contains(&"use_tool"),
            "MCP gateways stay off until a server connects: {:?}",
            names(&mutation)
        );
        let mut mcp_config = AgentConfig::default();
        mcp_config.memory.offer_mcp = true;
        let mcp_mutation = advertised_tools(
            &mcp_config,
            Some(("implement the parser", TaskIntent::Mutation)),
        );
        assert!(
            names(&mcp_mutation).contains(&"search_tool")
                && names(&mcp_mutation).contains(&"use_tool"),
            "connected MCP advertises search/select, not per-tool schemas: {:?}",
            names(&mcp_mutation)
        );
        let mut memory_config = AgentConfig::default();
        memory_config.memory.offer_memory = true;
        let memory_mutation = advertised_tools(
            &memory_config,
            Some(("implement the parser", TaskIntent::Mutation)),
        );
        for name in [
            "memory_search",
            "memory_get",
            "memory_update",
            "memory_forget",
        ] {
            assert!(
                names(&memory_mutation).contains(&name),
                "offer_memory should inject {name}: {:?}",
                names(&memory_mutation)
            );
        }
        assert!(
            !names(&mutation).contains(&"memory_search"),
            "memory tools stay off until attach_memory / offer_memory: {:?}",
            names(&mutation)
        );
        let memory_isolated = advertised_tools_with_background(
            &memory_config,
            Some((
                "Fix the bug in crates/hi-ai/src/openai/request.rs, then run the focused tests.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        assert!(
            !names(&memory_isolated).contains(&"memory_search")
                && !names(&memory_isolated).contains(&"memory_update"),
            "isolation catalog must not pay memory schema tax: {:?}",
            names(&memory_isolated)
        );
        let mcp_isolated = advertised_tools_with_background(
            &mcp_config,
            Some((
                "Fix the bug in crates/hi-ai/src/openai/request.rs, then run the focused tests.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        assert!(
            !names(&mcp_isolated).contains(&"search_tool")
                && !names(&mcp_isolated).contains(&"use_tool"),
            "isolation catalog must not pay MCP schema tax: {:?}",
            names(&mcp_isolated)
        );
        let browser_coding = advertised_tools(
            &config,
            Some(("implement the parser", TaskIntent::Mutation)),
        );
        assert!(
            !names(&browser_coding).contains(&"browser_exec"),
            "ordinary coding must not inject browser_exec: {:?}",
            names(&browser_coding)
        );
        let browser_ui = advertised_tools(
            &config,
            Some((
                "screenshot the login page and debug the CSS",
                TaskIntent::ReadOnly,
            )),
        );
        assert!(
            names(&browser_ui).contains(&"browser_exec"),
            "page/login/UI tasks inject browser_exec by default: {:?}",
            names(&browser_ui)
        );
        let mut browser_off = AgentConfig::default();
        browser_off.memory.offer_browser = false;
        let browser_off = advertised_tools(
            &browser_off,
            Some(("screenshot the login page", TaskIntent::ReadOnly)),
        );
        assert!(
            !names(&browser_off).contains(&"browser_exec"),
            "browser_exec stays off when [browser] enabled = false: {:?}",
            names(&browser_off)
        );
        let headless = advertised_tools_with_background(
            &config,
            Some(("screenshot the login page", TaskIntent::ReadOnly)),
            BackgroundToolAvailability {
                shell: true,
                tasks: true,
                interactive: false,
            },
        );
        assert!(
            !names(&headless).contains(&"browser_exec"),
            "browser_exec stays off without an interactive TTY: {:?}",
            names(&headless)
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
        assert!(
            !delegate_risk_relevant("Do not rewrite the parser"),
            "negated isolation verbs must not open the broad catalog"
        );
        assert!(delegate_risk_relevant("rewrite the parser"));
    }

    #[test]
    fn negated_rewrite_stays_a_targeted_named_mutation() {
        let prompt = "Write driver.py for the included host.py tool host.\n\
             Do not rewrite host.py or the oracle.\n\
             Do not edit bug/ yourself — only talk to host.py.";
        assert!(
            targeted_named_file_mutation(prompt, true),
            "path-scoped 'do not rewrite' must keep the lean named-file catalog"
        );
        assert!(
            !targeted_named_file_mutation("rewrite src/a.rs and src/b.rs in parallel", true),
            "an actual rewrite request must keep the broad catalog"
        );
        assert!(contains_unnegated("please rewrite host.py", "rewrite"));
        assert!(!contains_unnegated("do not rewrite host.py", "rewrite"));
        assert!(!contains_unnegated("don't migrate auth", "migrate"));
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

        let write_named = advertised_tools_with_background(
            &config,
            Some((
                "Write driver.py for the included host.py tool host.\n\
                 Do not rewrite host.py or the oracle.\n\
                 Do not edit bug/ yourself — only talk to host.py.",
                TaskIntent::Mutation,
            )),
            BackgroundToolAvailability::default(),
        );
        let write_named_names = names(&write_named);
        assert!(
            write_named_names.contains(&"write"),
            "write <filename> must advertise write: {write_named_names:?}"
        );
        assert!(
            !write_named_names.contains(&"explore"),
            "'do not rewrite' must not open the isolation catalog: {write_named_names:?}"
        );
        assert!(
            !write_named_names.contains(&"delegate"),
            "'do not rewrite' must not open the isolation catalog: {write_named_names:?}"
        );

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
            !review_names.contains(&"update_plan"),
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
                interactive: true,
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
                interactive: true,
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

    #[test]
    fn pipe_flash_dynamic_keeps_coding_core_and_drops_subagent_soup() {
        let mut config = AgentConfig::default();
        config.routing.model = "pipe/deepseek-v4-flash-0731".into();
        let task = (
            "implement a rust helper across the repository and run tests",
            TaskIntent::Mutation,
        );
        let tools = advertised_tools(&config, Some(task));
        let flash_names = names(&tools);
        for keep in [
            "read", "write", "edit", "bash", "grep", "glob", "list", "repo_map",
        ] {
            assert!(
                flash_names.contains(&keep),
                "missing {keep} in {flash_names:?}"
            );
        }
        for dropped in [
            "delegate",
            "task",
            "explore",
            "skill",
            "memory_search",
            "use_tool",
            "search_tool",
            "get_task_output",
            "wait_tasks",
            "kill_task",
        ] {
            assert!(
                !flash_names.contains(&dropped),
                "unexpected {dropped} in {flash_names:?}"
            );
        }

        config.memory.tool_set = ToolSet::Full;
        let full = advertised_tools(&config, Some(task));
        let full_names = names(&full);
        assert!(
            full_names.contains(&"explore") || full_names.contains(&"task"),
            "Full catalog should keep subagent tools: {full_names:?}"
        );

        config.memory.tool_set = ToolSet::Dynamic;
        config.routing.model = "pipe/glm-5.2".into();
        let glm = advertised_tools(&config, Some(task));
        let glm_names = names(&glm);
        assert!(
            glm_names.contains(&"explore") || glm_names.contains(&"task"),
            "GLM Dynamic catalog should stay unchanged: {glm_names:?}"
        );
    }
}
