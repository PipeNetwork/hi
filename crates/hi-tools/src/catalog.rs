//! Tool catalog: advertised specs, capability metadata, and classifiers.
//!
//! Pure data + classification — no I/O. Execute dispatch stays in [`crate::tools`].

use hi_ai::ToolSpec;
use serde_json::json;
use std::sync::LazyLock;

mod optional_specs;

pub use optional_specs::{
    ask_user_tool_spec, browser_exec_tool_spec, delegate_tool_spec, explore_tool_spec,
    get_task_output_tool_spec, kill_task_tool_spec, memory_forget_tool_spec, memory_get_tool_spec,
    memory_search_tool_spec, memory_update_tool_spec, new_context_tool_spec,
    research_read_tool_spec, research_tool_spec, search_tool_tool_spec, skill_tool_spec,
    task_tool_spec, use_tool_tool_spec, wait_tasks_tool_spec,
};

/// The tools advertised to the model each turn.
fn build_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "update_plan".into(),
            description: "Record or update a short task plan, shown to the user as a live checklist. Call it when starting a task that takes several steps — pass the full ordered list of steps — then call it again as you progress, ALWAYS passing the complete list with updated statuses (mark the step you're on `active`, finished steps `done`). Keep titles to a few words. Skip it for trivial one-step tasks.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "The full ordered list of plan steps, resubmitted in its entirety on every call.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string", "description": "Short description of the step (a few words)." },
                                "status": { "type": "string", "enum": ["pending", "active", "done"], "description": "pending (not started), active (in progress now), or done." }
                            },
                            "required": ["title", "status"]
                        }
                    }
                },
                "required": ["steps"]
            }),
        },
        ToolSpec {
            name: "record_decision".into(),
            description: "Record a key design decision so it persists across context compaction and keeps later turns consistent. Call this when you commit to an approach, a convention, or a non-obvious tradeoff (e.g. 'using a BTreeMap for ordered iteration', 'skipping Windows support for now'). Kept verbatim in the system prompt — NOT summarized away — so a long refactor doesn't drift from its own rationale. Use sparingly: only for decisions that matter later.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "summary": { "type": "string", "description": "A short title of the decision (one line)." },
                    "rationale": { "type": "string", "description": "Why this choice — the constraint or tradeoff that drove it." },
                    "files": {
                        "type": "array",
                        "description": "Files the decision most affects (may be empty).",
                        "items": { "type": "string" }
                    }
                },
                "required": ["summary", "rationale"]
            }),
        },
        ToolSpec {
            name: "block_step".into(),
            description: "Report that the active long-horizon goal step cannot be completed here because a prerequisite is missing from the environment — a service that isn't running, a binary that isn't installed, a credential that wasn't provided. Use this INSTEAD of retrying or writing a stub: retrying cannot install a database, and a stub that skips the required check is worse than an honest block. The step is set aside with your reason and the drive moves to the next one, so the user gets an actionable list. Only for missing prerequisites — if the work is merely hard, or you are unsure how to do it, keep working.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "prerequisite": {
                        "type": "string",
                        "description": "The specific missing thing, as concretely as you can name it (e.g. 'a running PostgreSQL reachable via DATABASE_URL', 'the `tofu` binary'). Name what to install or start, not what you tried."
                    }
                },
                "required": ["prerequisite"]
            }),
        },
        ToolSpec {
            name: "read".into(),
            description: "Read one or more UTF-8 text files. Lines are returned numbered (`<n>\\t<text>`). Each file is capped by the shared result budget; if the footer says `read more with offset N`, use that exact offset only when the missing lines are needed. For a summary, answer once the returned evidence is sufficient instead of paging automatically. Prefer this over `bash` `cat`/`sed`/`head` for source and spec files — those dumps are clipped and lose the middle. When a task names multiple files, use one call with the `paths` array instead of separate calls with `path`. For a named single-file edit, read that target first; do not read unrelated manifests or project files unless the requested change or validation actually needs them.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to read." },
                    "paths": {
                        "type": "array",
                        "description": "Multiple paths to read in one call; use instead of `path`.",
                        "minItems": 1,
                        "maxItems": 32,
                        "items": { "type": "string", "minLength": 1 }
                    },
                    "offset": { "type": "integer", "description": "1-based line to start at (default: first line)." },
                    "limit": { "type": "integer", "description": "Maximum number of lines to return (default: 2000)." }
                },
                "oneOf": [
                    { "required": ["path"] },
                    { "required": ["paths"] }
                ],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "write".into(),
            description: "Create a new file, or overwrite a small existing file, with the given content. Parent directories are created as needed. Do not use write to rewrite a large existing source file — use `edit` / `multi_edit` / `apply_patch` for in-place changes (large overwrites are rejected).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to write." },
                    "content": { "type": "string", "description": "Full content to write." }
                },
                "required": ["path", "content"]
            }),
        },
        ToolSpec {
            name: "edit".into(),
            description: "Replace a unique block of text in one file (preferred for ≤1 hunk on a known file). old_string must occur once and be the file's literal text WITHOUT the `read` line-number gutter; whitespace and indentation differences are tolerated. For independent edits across multiple files, make one edit call per file and batch those calls in one model turn when possible; never put paths for different files in one call. Set replace_all=true to replace every occurrence (use with care). On a miss, the tool re-reads once if the file changed underfoot.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to edit." },
                    "old_string": { "type": "string", "description": "Exact text to replace; must be unique in the file unless replace_all is set. Do not include line numbers." },
                    "new_string": { "type": "string", "description": "Replacement text." },
                    "replace_all": { "type": "boolean", "description": "If true, replace every occurrence of old_string (default: false, requires uniqueness)." }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        ToolSpec {
            name: "multi_edit".into(),
            description: "Apply several edits to one file atomically, in order. Each edit replaces a unique block (same rules as `edit`); if any fails, none are applied. Prefer this over multiple `edit` calls on the same file.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to edit." },
                    "edits": {
                        "type": "array",
                        "description": "Edits applied in sequence to the file's evolving content.",
                        "maxItems": 64,
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": { "type": "string", "description": "Exact text to replace; unique at the time this edit applies. No line numbers." },
                                "new_string": { "type": "string", "description": "Replacement text." }
                            },
                            "required": ["old_string", "new_string"]
                        }
                    }
                },
                "required": ["path", "edits"]
            }),
        },
        ToolSpec {
            name: "bash".into(),
            description: "Run a shell command via `sh -c` in the current working directory and return combined stdout/stderr. stdin is closed, so commands never block on input. A foreground command still running at its timeout continues in the background and returns a shell handle (`sh_N`) — read output with bash_output and stop with bash_kill. For a process you know upfront is long-lived or blocking (a dev server, a file watcher, `tail -f`), set run_in_background:true to get the handle immediately. For a slow but finite build or test suite, raise `timeout` so it finishes in the foreground. For very long background work (a big download, a multi-hour job), chain the follow-up steps into the command itself (`fetch && convert`) so nothing has to babysit it. On macOS/Linux, use `python3` rather than assuming a `python` command exists. Shell handles use the `sh_` prefix; agent subagent tasks use `task_` — do not mix them. Do not curl/wget a public http(s) URL the user already gave — use `web_fetch`. Do not use the shell as a search engine — use `web_search`. To inspect a workspace file, use the `read` tool instead of `cat`/`sed`/`head`.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to run." },
                    "timeout": { "type": "integer", "description": "Optional wall-clock limit in seconds (default 600, max 3600). Raise it for a slow test/build suite. Ignored when run_in_background is true." },
                    "run_in_background": { "type": "boolean", "description": "Run detached and return a handle id immediately instead of waiting for the command to exit. Use for servers/watchers/long-lived processes." }
                },
                "required": ["command"]
            }),
        },
        ToolSpec {
            name: "bash_output".into(),
            description: "Read new output (stdout+stderr) from a background shell started by `bash`, since the last read. Also reports whether it is still running, exited (with code), or was stopped. If the process is running with no new output yet, this call automatically waits for output or exit (with growing patience the quieter the process gets) before returning — so never re-poll in a loop; one call does the waiting. Pass wait_secs to set the patience yourself (max 600; 0 forces an instant peek). For work expected to outlast the turn (large downloads, long jobs), chain follow-up steps into the background command itself (`cmd && next`), report the current status, and stop instead of babysitting it.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The exact shell handle from a bash start message (a command-derived name like `cargo-test_3`). Only handles bash actually returned exist — never guess one. Not a task_ id." },
                    "wait_secs": { "type": "integer", "description": "Optional patience override: block up to this many seconds (max 600) for new output or exit. Omitted = automatic adaptive wait (recommended). 0 = instant non-blocking peek." }
                },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "bash_kill".into(),
            description: "Stop a background shell (and its whole process tree) started by `bash`. Idempotent. Pass the shell handle from the bash start message, not a task_ id.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The exact shell handle from a bash start message. Only handles bash actually returned exist — never guess one." }
                },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "list".into(),
            description: "List the project's files (respecting .gitignore), optionally under a subpath. Use this first to get the lay of the codebase before reading files.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory to list, relative to the project root (default: the whole project)." }
                }
            }),
        },
        ToolSpec {
            name: "diff".into(),
            description: "Show what's changed in the working tree versus the last commit (tracked changes as a diff, plus a list of new untracked files). Use this to review your own edits before finishing.".into(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolSpec {
            name: "grep".into(),
            description: "Search file contents for a regular expression (ripgrep if available, else grep), respecting .gitignore. Returns matching `path:line: text`. Use this to find where something is defined or used. Pass `context` to see surrounding lines. Pass `glob` to filter by file name pattern (e.g. `*.rs`).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Regular expression to search for." },
                    "path": { "type": "string", "description": "File or directory to search (default: the whole project)." },
                    "context": { "type": "integer", "description": "Lines of context to show around each match (default: 0)." },
                    "glob": { "type": "string", "description": "File name glob to filter (e.g. `*.rs`, `*.py`). Only files whose name matches are searched." }
                },
                "required": ["pattern"]
            }),
        },
        ToolSpec {
            name: "glob".into(),
            description: "Find files by name pattern (e.g. `**/*.rs`, `src/*.py`). Respects .gitignore. Returns matching paths, up to 500 results.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Glob pattern to match file paths (e.g. `**/*.rs`, `*.py`)." },
                    "path": { "type": "string", "description": "Directory to search in (default: the whole project)." }
                },
                "required": ["pattern"]
            }),
        },
        ToolSpec {
            name: "repo_map".into(),
            description: "Ranked repository map of important source files and their top-level declarations. Prefer this over blind `list` when orienting on a coding task. Optional `task` boosts path/symbol word hits; optional `path` scopes under a subdirectory.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "task": { "type": "string", "description": "Optional task text used to rank relevant files (identifiers and path words help)." },
                    "path": { "type": "string", "description": "Optional subdirectory to scope the map (project-relative)." },
                    "limit": { "type": "integer", "description": "Max files to return (default 40, max 100)." }
                }
            }),
        },
        ToolSpec {
            name: "find_symbol".into(),
            description: "Find definitions of a symbol by name across the repo (case-insensitive substring over fn/class/struct/trait/type/etc.). Prefer this over `grep` when you know the identifier. Returns `path` + line + kind.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Symbol name or fragment (e.g. WorkspaceRuntime, verify_password)." },
                    "path": { "type": "string", "description": "Optional subdirectory to scope the search (project-relative)." },
                    "limit": { "type": "integer", "description": "Max hits to return (default 24, max 100)." }
                },
                "required": ["query"]
            }),
        },
        ToolSpec {
            name: "apply_patch".into(),
            description: "Apply a multi-file (or multi-hunk) patch. Prefer `edit` for a single unique hunk in one file; use this for coordinated edits across several files. Format: '*** Begin Patch\\n*** Update File: path\\n@@ context @\\n-old\\n+new\\n unchanged\\n*** End Patch'. Also supports '*** Add File: path' and '*** Delete File: path'.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "patch": { "type": "string", "description": "The patch text in Begin/End Patch format." }
                },
                "required": ["patch"]
            }),
        },
        ToolSpec {
            name: "diagnostics".into(),
            description: "Get LSP diagnostics (errors/warnings) for a file. Requires `/lsp on`. Returns line-level errors — cheaper and more precise than running a full build. Empty path returns diagnostics for all open files.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file (relative to cwd)." }
                },
                "required": []
            }),
        },
        ToolSpec {
            name: "definition".into(),
            description: "Goto definition of the symbol at a position. Requires `/lsp on`. Returns file:line:col locations. More precise than grep — respects scopes and types.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file." },
                    "line": { "type": "integer", "description": "0-based line number." },
                    "column": { "type": "integer", "description": "0-based character offset." }
                },
                "required": ["path", "line", "column"]
            }),
        },
        ToolSpec {
            name: "references".into(),
            description: "Find all references to the symbol at a position. Requires `/lsp on`. Returns call sites as file:line:col. Semantically correct — no false matches from comments or strings.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file." },
                    "line": { "type": "integer", "description": "0-based line number." },
                    "column": { "type": "integer", "description": "0-based character offset." }
                },
                "required": ["path", "line", "column"]
            }),
        },
        ToolSpec {
            name: "hover".into(),
            description: "Get type and documentation for the symbol at a position. Requires `/lsp on`. Returns the hover text (type signature, docs).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file." },
                    "line": { "type": "integer", "description": "0-based line number." },
                    "column": { "type": "integer", "description": "0-based character offset." }
                },
                "required": ["path", "line", "column"]
            }),
        },
        ToolSpec {
            name: "web_search".into(),
            description: "Search the web for current information outside the repo — library docs, API specs, current events, model catalogs, recent release notes. Returns cited results (title, URL, snippet). Don't use this for things `read`/`grep`/`list` can answer locally. If the user already gave an exact http(s) URL, use `web_fetch` instead of searching.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query." },
                    "max_results": { "type": "integer", "description": "Maximum results to return (default 5, cap 10)." }
                },
                "required": ["query"]
            }),
        },
        ToolSpec {
            name: "web_fetch".into(),
            description: "Fetch a public URL and return its content (JSON pretty-printed, HTML stripped to text, truncated). No API key needed. Use this for documentation pages, public API URLs, or any direct URL the model needs to read — prefer it over `bash` curl/wget. For search-engine results use `web_search`; for Hugging Face model discovery use `/hf` or Hub API URLs explicitly.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The http:// or https:// URL to fetch." }
                },
                "required": ["url"]
            }),
        },
        ToolSpec {
            name: "web_download".into(),
            description: "Download a file from Hugging Face Hub or any direct public URL. Runs in the background — returns a handle to poll with `bash_output` and stop with `bash_kill`. For a Hugging Face repo, pass `source` as `org/model`, `org/model@revision`, or `org/model@revision:filename`; if no filename is given, lists the repo's files first. Full HTTP(S) URLs are direct downloads, not Hub discovery. The `output` path defaults to the file's basename and must be within the workspace.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "source": { "type": "string", "description": "Hugging Face repo ref (`org/model`, `org/model@revision`, `org/model:filename`) or full URL." },
                    "filename": { "type": "string", "description": "Filename within the repo (optional — if omitted, lists available files)." },
                    "output": { "type": "string", "description": "Local path to save the file (defaults to basename, must be in workspace)." }
                },
                "required": ["source"]
            }),
        },
    ]
}

/// The tool specifications advertised to the model, cached once.
pub static TOOL_SPECS: LazyLock<Vec<ToolSpec>> = LazyLock::new(build_tool_specs);

/// The core workspace loop — tools that census-driven trimming
/// (`hi tools trim`) must never remove, no matter what the usage data says.
/// A model without read/edit/bash is not a leaner agent, it is a broken one;
/// keeping the floor here (and enforcing it again at advertisement time)
/// makes a wrong or corrupted trim list harmless.
///
/// Almost never expand this list for new tools — see
/// `docs/adr/002-tool-admission.md`. New capabilities default to inject or
/// capability-gated ads, or to bash/skills when a thin CLI wrapper would do.
pub const PROTECTED_TOOLS: &[&str] = &[
    "read",
    "write",
    "edit",
    "multi_edit",
    "bash",
    "grep",
    "list",
    "diff",
];

/// Capability family used for task-aware tool advertisement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCapability {
    Coordination,
    Repository,
    Mutation,
    Process,
    Background,
    Lsp,
    Web,
    Subagent,
    Mcp,
    Memory,
    Skill,
}

/// Why a tool is first-class instead of bash/skill — see
/// `docs/adr/002-tool-admission.md`. Every catalog row must pick one of
/// these three gates; there is no grandfather variant because the existing
/// set is already fully classified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolAdmission {
    /// Parseable transcript results beat free-form CLI noise.
    Structure,
    /// Confirmations, sandbox/side-effect class, or control-plane needs.
    Safety,
    /// Materially more reliable than the raw human path on real turns.
    Reliability,
}

/// Authoritative behavioral metadata for every built-in and injected tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolMetadata {
    pub name: &'static str,
    pub capability: ToolCapability,
    pub read_only: bool,
    pub filesystem_mutating: bool,
    pub minimal: bool,
    /// Admission gate that justifies a first-class tool (ADR 002).
    pub admission: ToolAdmission,
    /// Human-protocol alternative considered (bash/CLI/skill/editor).
    pub alternative: &'static str,
}

macro_rules! tool_metadata {
    (
        $name:literal,
        $capability:ident,
        $read_only:literal,
        $mutating:literal,
        $minimal:literal,
        $admission:ident,
        $alternative:literal
    ) => {
        ToolMetadata {
            name: $name,
            capability: ToolCapability::$capability,
            read_only: $read_only,
            filesystem_mutating: $mutating,
            minimal: $minimal,
            admission: ToolAdmission::$admission,
            alternative: $alternative,
        }
    };
}

pub const TOOL_CATALOG: &[ToolMetadata] = &[
    tool_metadata!(
        "update_plan",
        Coordination,
        true,
        false,
        true,
        Structure,
        "free-text plan in the reply"
    ),
    tool_metadata!(
        "record_decision",
        Coordination,
        true,
        false,
        false,
        Structure,
        "note in session memory markdown"
    ),
    tool_metadata!(
        "block_step",
        Coordination,
        true,
        false,
        false,
        Structure,
        "update_plan + prose status"
    ),
    tool_metadata!(
        "ask_user",
        Coordination,
        true,
        false,
        false,
        Structure,
        "ask in assistant text (cannot pause the turn)"
    ),
    tool_metadata!(
        "new_context",
        Coordination,
        true,
        false,
        false,
        Structure,
        "/window or /compact (those summarize or require the user)"
    ),
    tool_metadata!(
        "read",
        Repository,
        true,
        false,
        true,
        Reliability,
        "bash cat/sed with line noise"
    ),
    tool_metadata!(
        "write",
        Mutation,
        false,
        true,
        true,
        Safety,
        "bash redirection without confirm/checkpoint"
    ),
    tool_metadata!(
        "edit",
        Mutation,
        false,
        true,
        true,
        Reliability,
        "bash sed/heredoc partial applies"
    ),
    tool_metadata!(
        "multi_edit",
        Mutation,
        false,
        true,
        false,
        Reliability,
        "repeated edit or sed scripts"
    ),
    tool_metadata!(
        "bash",
        Process,
        false,
        false,
        true,
        Safety,
        "no alternative — human-protocol escape hatch"
    ),
    // `bash` can return a background handle even when it starts in the
    // foreground. Keep its poll/stop controls in the minimal catalog so the
    // model is never instructed to call a tool whose schema was withheld.
    tool_metadata!(
        "bash_output",
        Background,
        true,
        false,
        true,
        Structure,
        "blocking bash only"
    ),
    tool_metadata!(
        "bash_kill",
        Background,
        false,
        false,
        true,
        Safety,
        "bash kill without handle tracking"
    ),
    tool_metadata!("list", Repository, true, false, true, Structure, "bash ls"),
    tool_metadata!(
        "diff",
        Repository,
        true,
        false,
        false,
        Structure,
        "bash git diff"
    ),
    tool_metadata!(
        "grep",
        Repository,
        true,
        false,
        true,
        Structure,
        "bash rg/grep"
    ),
    tool_metadata!(
        "glob",
        Repository,
        true,
        false,
        true,
        Structure,
        "bash find"
    ),
    tool_metadata!(
        "repo_map",
        Repository,
        true,
        false,
        true,
        Structure,
        "blind list/grep orientation"
    ),
    tool_metadata!(
        "find_symbol",
        Repository,
        true,
        false,
        true,
        Structure,
        "bash rg for definitions"
    ),
    tool_metadata!(
        "apply_patch",
        Mutation,
        false,
        true,
        false,
        Reliability,
        "multi-file bash patches"
    ),
    tool_metadata!(
        "diagnostics",
        Lsp,
        true,
        false,
        false,
        Structure,
        "bash cargo check/tsc with unstructured noise"
    ),
    tool_metadata!(
        "definition",
        Lsp,
        true,
        false,
        false,
        Structure,
        "bash rg/ctags"
    ),
    tool_metadata!("references", Lsp, true, false, false, Structure, "bash rg"),
    tool_metadata!(
        "hover",
        Lsp,
        true,
        false,
        false,
        Structure,
        "read source + docs in browser"
    ),
    tool_metadata!(
        "web_search",
        Web,
        true,
        false,
        false,
        Structure,
        "bash curl to a search API"
    ),
    tool_metadata!(
        "web_fetch",
        Web,
        true,
        false,
        false,
        Structure,
        "bash curl | html2text"
    ),
    tool_metadata!(
        "research",
        Web,
        true,
        false,
        false,
        Structure,
        "web_search + web_fetch without rerank"
    ),
    tool_metadata!(
        "research_read",
        Web,
        true,
        false,
        false,
        Reliability,
        "web_fetch of a researched page_id"
    ),
    tool_metadata!(
        "web_download",
        Web,
        false,
        true,
        false,
        Safety,
        "bash curl -O without workspace confine"
    ),
    tool_metadata!(
        "explore",
        Subagent,
        false,
        false,
        false,
        Reliability,
        "serial read/grep in parent context"
    ),
    tool_metadata!(
        "delegate",
        Subagent,
        false,
        false,
        false,
        Reliability,
        "manual worktree + second hi process"
    ),
    tool_metadata!(
        "task",
        Subagent,
        false,
        false,
        false,
        Structure,
        "delegate only (no async handle)"
    ),
    tool_metadata!(
        "get_task_output",
        Subagent,
        true,
        false,
        false,
        Structure,
        "blocking task completion only"
    ),
    tool_metadata!(
        "wait_tasks",
        Subagent,
        true,
        false,
        false,
        Structure,
        "polling get_task_output"
    ),
    tool_metadata!(
        "kill_task",
        Subagent,
        false,
        false,
        false,
        Safety,
        "OS kill without task registry"
    ),
    tool_metadata!(
        "use_tool",
        Mcp,
        false,
        false,
        false,
        Structure,
        "bash against each external CLI (when one exists)"
    ),
    tool_metadata!(
        "search_tool",
        Mcp,
        true,
        false,
        false,
        Structure,
        "read MCP server docs manually"
    ),
    tool_metadata!(
        "memory_search",
        Memory,
        true,
        false,
        false,
        Structure,
        "grep session memory files"
    ),
    tool_metadata!(
        "memory_get",
        Memory,
        true,
        false,
        false,
        Structure,
        "read memory files with read"
    ),
    tool_metadata!(
        "memory_update",
        Memory,
        false,
        true,
        false,
        Structure,
        "/remember or edit .hi/memory.md"
    ),
    tool_metadata!(
        "memory_forget",
        Memory,
        false,
        true,
        false,
        Structure,
        "/undo-memory or edit .hi/memory.md"
    ),
    tool_metadata!(
        "skill",
        Skill,
        true,
        false,
        false,
        Structure,
        "/skill slash or read SKILL.md"
    ),
    tool_metadata!(
        "browser_exec",
        Web,
        false,
        false,
        false,
        Safety,
        "bash curl/wget or a browser skill"
    ),
];

pub fn tool_metadata(name: &str) -> Option<&'static ToolMetadata> {
    TOOL_CATALOG.iter().find(|metadata| metadata.name == name)
}

pub fn is_known_tool(name: &str) -> bool {
    tool_metadata(name).is_some()
}

/// Essential tools kept for small models. A model around 3B can't reliably plan
/// over the full ~20-tool set — the large, detailed tool schema degrades its
/// structured-output quality and latency sharply (empirically, tool-calling
/// slowed ~15x from 6 tools to 21 and eventually produced malformed calls). This
/// lean file-navigation + edit + shell set keeps such models usable.
pub static MINIMAL_TOOL_SPECS: LazyLock<Vec<ToolSpec>> = LazyLock::new(|| {
    TOOL_SPECS
        .iter()
        .filter(|spec| tool_metadata(&spec.name).is_some_and(|metadata| metadata.minimal))
        .cloned()
        .collect()
});

/// Whether a tool only observes state, with no side effects — so several can
/// run concurrently within one round, and it's safe to offer in `ReadOnly`
/// tool mode. Tools that mutate the filesystem (`write`, `edit`, `multi_edit`,
/// `apply_patch`) or have ordering-sensitive external effects (`bash`,
/// `bash_kill`) are excluded. `update_plan` and `record_decision` have no
/// side effects beyond in-memory state, so they're read-only here.
/// `bash_output` is a pure poll of an existing buffer.
pub fn is_read_only(name: &str) -> bool {
    tool_metadata(name).is_some_and(|metadata| metadata.read_only)
}

/// Whether a tool mutates the working tree — so the agent should invalidate its
/// snapshot cache and kick off a proactive fast-check after it runs. This is a
/// narrower set than `!is_read_only`: `bash` can mutate files but is handled
/// separately (it always runs alone), and `bash_kill`/`update_plan`/
/// `record_decision` have no filesystem effect even though they're not
/// read-only for parallelization purposes.
pub fn is_filesystem_mutating(name: &str) -> bool {
    tool_metadata(name).is_some_and(|metadata| metadata.filesystem_mutating)
}

/// Whether a tool is pure bookkeeping (`update_plan`, `record_decision`):
/// it records agent-side coordination state and does no work on the task
/// itself. The agent's steering uses this to spot rounds that only shuffle
/// bookkeeping — a weak-model stall pattern — and to withhold these tools for
/// a round when the model fixates on them.
pub fn is_coordination(name: &str) -> bool {
    tool_metadata(name).is_some_and(|metadata| metadata.capability == ToolCapability::Coordination)
}

/// Best-effort extraction of the primary target path from a tool call's JSON
/// arguments — the `path` field for read/write/edit/list, the `path`/`glob` for
/// grep. Returns `None` for tools without a meaningful single path (e.g.
/// `bash`, or a `grep` with only a pattern). Used by the agent to infer
/// within-batch dependencies: a read of a file a mutating call earlier in the
/// same batch targeted should observe that mutation, so it's serialized after.
/// Tolerant — a failed parse yields `None`, which the caller treats as "no
/// dependency inferred" (safe fallback to emission order).
pub fn target_path(name: &str, arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    match name {
        // read/write/edit/multi_edit carry an explicit `path`. `read` may also
        // use `paths` (an array): a one-element array is that single path; a
        // multi-element array has no single target, so return None and let
        // dependency inference treat it conservatively.
        "read" => value
            .get("path")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| {
                value.get("paths").and_then(|v| v.as_array()).and_then(|a| {
                    if a.len() == 1 {
                        a[0].as_str().map(str::to_string)
                    } else {
                        None
                    }
                })
            }),
        "write" | "edit" | "multi_edit" => value.get("path")?.as_str().map(str::to_string),
        // list's path is optional (defaults to ".").
        "list" => value.get("path")?.as_str().map(str::to_string),
        // Optional scope path for orientation tools (directory, not a single file).
        "repo_map" | "find_symbol" => value.get("path")?.as_str().map(str::to_string),
        // grep: prefer an explicit `path`; fall back to `glob` only as a hint
        // (a glob isn't a single file, so return None to avoid over-serializing).
        "grep" => value.get("path")?.as_str().map(str::to_string),
        // apply_patch: the patch text contains `*** Update File: <path>` (or
        // `*** Add File:`/`*** Delete File:`) directives. Return the path only
        // when the patch targets exactly one file. Multi-file patches have no
        // single target, so return None and let dependency inference treat the
        // mutation as unknown-path, serializing later reads conservatively.
        "apply_patch" => {
            let patch = value.get("patch")?.as_str()?;
            let mut paths: Vec<String> = patch
                .lines()
                .filter_map(|line| {
                    line.trim()
                        .strip_prefix("*** Update File: ")
                        .or_else(|| line.trim().strip_prefix("*** Add File: "))
                        .or_else(|| line.trim().strip_prefix("*** Delete File: "))
                        .map(str::trim)
                        .filter(|path| !path.is_empty())
                        .map(str::to_string)
                })
                .collect();
            paths.sort();
            paths.dedup();
            if paths.len() == 1 { paths.pop() } else { None }
        }
        // diff/glob/bash: no single meaningful target path for dep inference.
        _ => None,
    }
}

/// Every concrete path this call targeted. Used by the eval tape: a batched
/// `read` of several files must still count as a read of each one.
pub fn target_paths(name: &str, arguments: &str) -> Vec<String> {
    if let Some(one) = target_path(name, arguments) {
        return vec![one];
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return Vec::new();
    };
    if name == "read" {
        return value
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .filter(|path| !path.is_empty())
                    .collect()
            })
            .unwrap_or_default();
    }
    Vec::new()
}

#[cfg(test)]
mod tests;
