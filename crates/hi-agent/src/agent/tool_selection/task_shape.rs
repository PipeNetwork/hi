//! Prompt-shape classification for dynamic tool advertisement.

use std::collections::BTreeSet;

use crate::{AgentConfig, TaskIntent, ToolSet, WriteSubagentPolicy, steering::is_file_reference};

/// Whether the prompt explicitly requests one list operation and nothing that
/// needs a second repository tool. This is intentionally stricter than the
/// general dynamic catalog: a vague "list the issues" or a dependent
/// "list, then read" request must retain the broader catalog so the model can
/// continue without receiving an unknown-tool error.
pub(super) fn direct_list_task(task: &str, mutating: bool) -> bool {
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
pub(super) fn direct_list_read_sequence_task(task: &str, mutating: bool) -> bool {
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

pub(super) fn should_advertise_delegate(
    config: &AgentConfig,
    task: Option<&str>,
    mutating: bool,
) -> bool {
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
pub(super) fn direct_file_summary_task(task: &str, mutating: bool) -> bool {
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
pub(super) fn targeted_mutation_file_count(task: &str, mutating: bool) -> Option<usize> {
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
    .any(|marker| contains_unnegated(&lower, marker))
        || contains_unnegated_word(&lower, "port");
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
    .any(|marker| contains_unnegated(&lower, marker));
    let file_mentions = lower
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-' | '/'))
        })
        .filter(|token| is_file_reference(token))
        .collect::<BTreeSet<_>>();
    (explicit && !isolation && !broad).then_some(file_mentions.len())
}

/// Whether a prompt is an explicit mutation of at most four named files.
/// Used to skip the repository index: the prompt already named the files.
pub(in crate::agent) fn targeted_named_file_mutation(task: &str, mutating: bool) -> bool {
    matches!(targeted_mutation_file_count(task, mutating), Some(1..=4))
}

/// Whether a prompt is an explicit, single-file mutation.
pub(super) fn targeted_single_file_mutation_task(task: &str, mutating: bool) -> bool {
    targeted_mutation_file_count(task, mutating) == Some(1)
}

/// Whether a prompt is an explicit, small multi-file mutation. Larger or
/// isolation-shaped changes retain the broad catalog so the model can plan,
/// search, or delegate when that is actually useful.
pub(super) fn targeted_multi_file_mutation_task(task: &str, mutating: bool) -> bool {
    matches!(targeted_mutation_file_count(task, mutating), Some(2..=4))
}

/// True when `needle` occurs without a negation immediately before it.
/// "Do not rewrite host.py" must not count as isolation-shaped `rewrite`.
pub(super) fn contains_unnegated(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let mut from = 0;
    while from < haystack.len() {
        let Some(rel) = haystack[from..].find(needle) else {
            return false;
        };
        let start = from + rel;
        if !negated_at(haystack, start) {
            return true;
        }
        from = start + needle.len();
    }
    false
}

pub(super) fn contains_unnegated_word(haystack: &str, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    let mut from = 0;
    while from < haystack.len() {
        let Some(rel) = haystack[from..].find(word) else {
            return false;
        };
        let start = from + rel;
        let end = start + word.len();
        let before_ok = start == 0 || !haystack.as_bytes()[start - 1].is_ascii_alphanumeric();
        let after_ok = end == haystack.len()
            || haystack
                .as_bytes()
                .get(end)
                .is_some_and(|byte| !byte.is_ascii_alphanumeric());
        if before_ok && after_ok && !negated_at(haystack, start) {
            return true;
        }
        from = end.max(from + 1);
    }
    false
}

pub(super) fn negated_at(haystack: &str, start: usize) -> bool {
    let before = haystack[..start].trim_end();
    let Some(token) = before.split_whitespace().next_back() else {
        return false;
    };
    let token = token.trim_matches(|character: char| {
        matches!(
            character,
            ',' | ';' | ':' | '"' | '\'' | '(' | ')' | '!' | '?' | '.' | '—' | '–'
        )
    });
    matches!(
        token,
        "not" | "never" | "without" | "avoid" | "don't" | "dont" | "no" | "can't" | "cannot"
    )
}

/// Search is an optional branch for a known-file edit. Keep its schema out of
/// the common path unless the prompt explicitly asks for review/search work;
/// the model can inspect the named file directly with `read`.
pub(super) fn targeted_mutation_needs_search(task: &str) -> bool {
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
pub(super) fn targeted_mutation_needs_diff(task: &str) -> bool {
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
pub(super) fn targeted_mutation_needs_plan(task: &str) -> bool {
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
pub(super) fn targeted_mutation_needs_write(task: &str) -> bool {
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
        || write_named_file_request(&lower)
}

/// "Write driver.py" / "write src/foo.rs" names a new file without using the
/// longer "write a file" phrasing the marker list above expects.
pub(super) fn write_named_file_request(lower: &str) -> bool {
    let words: Vec<&str> = lower
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || matches!(character, '.' | '/' | '_' | '-'))
        })
        .filter(|word| !word.is_empty())
        .collect();
    words.windows(2).any(|pair| {
        matches!(pair[0], "write" | "writing")
            && (is_file_reference(pair[1]) || pair[1].contains('/'))
    })
}

/// An exact `http(s)://…` in the prompt is a fetch, not a search. The word
/// "https" alone (as in "write an https URL") does not count.
pub(super) fn prompt_has_concrete_http_url(lower: &str) -> bool {
    lower.contains("http://") || lower.contains("https://")
}

/// `write answer.txt` after fetching a given URL still looks like a targeted
/// one-file mutation. Keep `web_fetch` or the model only has `bash` curl.
pub(super) fn targeted_mutation_needs_web_fetch(task: &str) -> bool {
    prompt_has_concrete_http_url(&task.to_ascii_lowercase())
}

/// Research / "on the web" writes are the same shape. Keep `web_search` and
/// leave `web_fetch` out so the model does not guess a URL first. The word
/// "web" in "do not use the web" must not open this schema.
pub(super) fn targeted_mutation_needs_web_search(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    if prompt_has_concrete_http_url(&lower) {
        return false;
    }
    [
        "search the web",
        "web search",
        "on the web",
        "research ",
        "look up online",
        "documentation online",
    ]
    .iter()
    .any(|marker| contains_unnegated(&lower, marker))
}

/// Inject `browser_exec` for page/login/live-UI work, not ordinary coding.
pub(super) fn should_advertise_browser(task: Option<&str>) -> bool {
    let Some(task) = task else {
        return false;
    };
    let lower = task.to_ascii_lowercase();
    if lower.contains("http://") || lower.contains("https://") {
        return true;
    }
    [
        "login",
        "screenshot",
        "headless",
        "browser",
        "css",
        "dom",
        "devtools",
    ]
    .iter()
    .any(|word| contains_unnegated_word(&lower, word))
        || [
            "the page",
            "web page",
            "webpage",
            "debug the ui",
            "debug ui",
            "login form",
            "sign-in",
            "sign in page",
            "accessibility tree",
            "live ui",
        ]
        .iter()
        .any(|marker| contains_unnegated(&lower, marker))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResearchInjection {
    None,
    ReadOnly,
    SearchAndRead,
}

/// Inject-only Pipe research tools. RSI managed `code.change` workers skip
/// them even when the lease allowlists the names (cost / nondeterminism).
pub(super) fn research_injection(config: &AgentConfig, task: Option<&str>) -> ResearchInjection {
    if config.rsi.managed {
        return ResearchInjection::None;
    }
    let snippets_injected = std::env::var("HI_RESEARCH_SNIPPETS_INJECTED")
        .ok()
        .is_some_and(|value| value == "1");
    if snippets_injected {
        return ResearchInjection::ReadOnly;
    }
    if !hi_tools::research_credentials_configured() {
        return ResearchInjection::None;
    }
    let Some(task) = task else {
        return ResearchInjection::None;
    };
    if targeted_mutation_needs_web_search(task) {
        ResearchInjection::SearchAndRead
    } else {
        ResearchInjection::None
    }
}

/// Do not advertise a process tool when the user explicitly forbids shell
/// validation. Keeping the schema out of the request makes that constraint
/// enforceable instead of relying on the model to ignore an available tool.
pub(super) fn targeted_mutation_allows_shell(task: &str) -> bool {
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
    .any(|marker| contains_unnegated(&lower, marker))
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
    let has_port_word = contains_unnegated_word(&lower, "port");
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
        "rewrite",
        "split into",
        "extract into",
    ]
    .iter()
    .any(|marker| contains_unnegated(&lower, marker))
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

pub(super) fn repository_tools_relevant(task: &str, intent: TaskIntent) -> bool {
    if intent == TaskIntent::ReadOnly
        && crate::task_contract::prompt_requests_exact_text_response(task)
    {
        return false;
    }
    let lower = task.to_ascii_lowercase();
    intent == TaskIntent::Mutation
        || explicitly_repository_relevant(&lower)
        || (!externally_scoped(&lower) && !clearly_conversational(&lower))
}

pub(super) fn broad_read_only_review(task: &str, mutating: bool) -> bool {
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

pub(super) fn explicit_subagent_request(task: &str) -> bool {
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

pub(super) fn externally_scoped(lower: &str) -> bool {
    lower.contains("http://")
        || lower.contains("https://")
        || lower
            .split(|character: char| !character.is_ascii_alphanumeric())
            .any(|word| matches!(word, "internet" | "online" | "web"))
}

pub(super) fn web_relevant(lower: &str) -> bool {
    externally_scoped(lower)
        || ["latest", "current", "release notes", "documentation"]
            .iter()
            .any(|marker| lower.contains(marker))
}

pub(super) fn explicitly_repository_relevant(lower: &str) -> bool {
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

pub(super) fn clearly_conversational(lower: &str) -> bool {
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
