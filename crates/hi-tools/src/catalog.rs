//! Tool catalog: advertised specs, capability metadata, and classifiers.
//!
//! Pure data + classification — no I/O. Execute dispatch stays in [`crate::tools`].

use hi_ai::ToolSpec;
use serde_json::json;
use std::sync::LazyLock;

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
            description: "Read a UTF-8 text file. Lines are returned numbered (`<n>\\t<text>`). Returns at most 2000 lines by default (the whole file for most source files); page with offset/limit instead of assuming you saw everything.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to read." },
                    "offset": { "type": "integer", "description": "1-based line to start at (default: first line)." },
                    "limit": { "type": "integer", "description": "Maximum number of lines to return (default: 2000)." }
                },
                "required": ["path"]
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
            description: "Replace a unique block of text in a file (preferred for ≤1 hunk on a known file). old_string must occur once and be the file's literal text WITHOUT the `read` line-number gutter; whitespace and indentation differences are tolerated. Set replace_all=true to replace every occurrence (use with care). On a miss, the tool re-reads once if the file changed underfoot.".into(),
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
            description: "Run a shell command via `sh -c` in the current working directory and return combined stdout/stderr. stdin is closed, so commands never block on input. A foreground command still running at its timeout is moved to the background (kept running, not killed) and returns a handle id — read its output with bash_output and stop it with bash_kill. For a process you know upfront is long-lived or blocking (a dev server, a file watcher, `tail -f`), set run_in_background:true to get the handle immediately. For a slow but finite build or test suite, raise `timeout` so it finishes in the foreground.".into(),
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
            description: "Read new output (stdout+stderr) from a background process started by `bash` with run_in_background, since the last read. Also reports whether it is still running, exited (with code), or was killed. Returns immediately. Do not tight-poll while it reports running with no new output — sleep meaningfully between checks, do other work, or for a finite build/test raise `bash` timeout and run it in the foreground instead.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The background process handle returned by bash (e.g. `bg_1`)." }
                },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "bash_kill".into(),
            description: "Stop a background process (and its whole process tree) started by `bash` with run_in_background. Idempotent.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The background process handle to kill (e.g. `bg_1`)." }
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
            description: "Search the web for current information outside the repo — library docs, API specs, current events, model catalogs, recent release notes. Returns cited results (title, URL, snippet). Don't use this for things `read`/`grep`/`list` can answer locally.".into(),
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
            description: "Fetch a public URL and return its content (JSON pretty-printed, HTML stripped to text, truncated). No API key needed. Use this for documentation pages, public API URLs, or any direct URL the model needs to read. For search-engine results use `web_search`; for Hugging Face model discovery use `/hf` or Hub API URLs explicitly.".into(),
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

/// Authoritative behavioral metadata for every built-in and injected tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolMetadata {
    pub name: &'static str,
    pub capability: ToolCapability,
    pub read_only: bool,
    pub filesystem_mutating: bool,
    pub minimal: bool,
}

macro_rules! tool_metadata {
    ($name:literal, $capability:ident, $read_only:literal, $mutating:literal, $minimal:literal) => {
        ToolMetadata {
            name: $name,
            capability: ToolCapability::$capability,
            read_only: $read_only,
            filesystem_mutating: $mutating,
            minimal: $minimal,
        }
    };
}

pub const TOOL_CATALOG: &[ToolMetadata] = &[
    tool_metadata!("update_plan", Coordination, true, false, true),
    tool_metadata!("record_decision", Coordination, true, false, false),
    tool_metadata!("block_step", Coordination, true, false, false),
    tool_metadata!("read", Repository, true, false, true),
    tool_metadata!("write", Mutation, false, true, true),
    tool_metadata!("edit", Mutation, false, true, true),
    tool_metadata!("multi_edit", Mutation, false, true, false),
    tool_metadata!("bash", Process, false, false, true),
    tool_metadata!("bash_output", Background, true, false, false),
    tool_metadata!("bash_kill", Background, false, false, false),
    tool_metadata!("list", Repository, true, false, true),
    tool_metadata!("diff", Repository, true, false, false),
    tool_metadata!("grep", Repository, true, false, true),
    tool_metadata!("glob", Repository, true, false, true),
    tool_metadata!("repo_map", Repository, true, false, true),
    tool_metadata!("find_symbol", Repository, true, false, true),
    tool_metadata!("apply_patch", Mutation, false, true, false),
    tool_metadata!("diagnostics", Lsp, true, false, false),
    tool_metadata!("definition", Lsp, true, false, false),
    tool_metadata!("references", Lsp, true, false, false),
    tool_metadata!("hover", Lsp, true, false, false),
    tool_metadata!("web_search", Web, true, false, false),
    tool_metadata!("web_fetch", Web, true, false, false),
    tool_metadata!("web_download", Web, false, true, false),
    tool_metadata!("explore", Subagent, false, false, false),
    tool_metadata!("delegate", Subagent, false, false, false),
    tool_metadata!("task", Subagent, false, false, false),
    tool_metadata!("get_task_output", Subagent, true, false, false),
    tool_metadata!("wait_tasks", Subagent, true, false, false),
    tool_metadata!("kill_task", Subagent, false, false, false),
    tool_metadata!("use_tool", Mcp, false, false, false),
    tool_metadata!("search_tool", Mcp, true, false, false),
    tool_metadata!("memory_search", Memory, true, false, false),
    tool_metadata!("memory_get", Memory, true, false, false),
    tool_metadata!("skill", Skill, true, false, false),
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

/// The `explore` read-only subagent tool. Deliberately kept OUT of [`TOOL_SPECS`]
/// and out of [`is_read_only`]: it's only advertised when the agent explicitly
/// injects it (for a capable parent via `explore_subagents`), and because it's not
/// read-only it never survives into a `ReadOnly` child's tool set — so a subagent
/// cannot spawn another (depth is capped at 1 structurally).
pub fn explore_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "explore".into(),
        description: "Delegate a focused, READ-ONLY investigation to a subagent that runs in its own fresh context and returns just a concise answer. Use it to keep your own context clean when a question needs reading or searching across many files — e.g. \"where is X configured and how is it used?\", \"summarize how module Y works\", \"find every call site of Z and what each passes\". The subagent can only read/list/grep/glob and inspect code (no edits, no shell, no spawning). Give it ONE self-contained task with enough detail to answer standalone. Prefer it over reading many files yourself when you only need the conclusion; don't use it for trivial single-file lookups or anything that must change files.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A single, self-contained read-only investigation to carry out, with enough context to answer on its own. Be specific about what to find and what to report back."
                }
            },
            "required": ["task"]
        }),
    }
}

/// The `delegate` write-capable subagent tool. Like [`explore_tool_spec`] it's kept
/// OUT of [`TOOL_SPECS`] and [`is_read_only`], and is only injected for a top-level
/// agent (via `write_subagents`) — never for a subagent, so it can't recurse.
pub fn delegate_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "delegate".into(),
        description: "Delegate a self-contained IMPLEMENTATION subtask to a subagent that runs in its own fresh context, can edit files and run commands, and verifies its own work. Its changes are merged back into your working tree ONLY if verification passes — otherwise they're rolled back automatically. Use it to hand off a well-scoped, independent chunk of work (e.g. \"implement the FooBar parser in src/foo.rs so `cargo test foo` passes\", \"add input validation to the signup handler and update its tests\") so it stays out of your context. Give ONE self-contained task with enough detail to complete standalone, and include how success is checked. Prefer doing small edits yourself; use this for a substantial, independently-verifiable subtask. The subagent cannot itself delegate or explore.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A single, self-contained implementation subtask with enough detail to complete standalone, including what 'done' looks like."
                },
                "verify": {
                    "type": "string",
                    "description": "Optional shell command that must pass for the subagent's changes to be kept (e.g. `cargo test foo`). If omitted, the session's verify command is used."
                }
            },
            "required": ["task"]
        }),
    }
}

/// The `task` tool — spawns a background subagent (explore or delegate) that runs
/// asynchronously while the parent continues working. Returns immediately with a
/// task handle; poll results with `get_task_output` or `wait_tasks`, cancel with
/// `kill_task`. Like `explore`/`delegate`, kept OUT of `TOOL_SPECS` and injected
/// only for a top-level agent.
pub fn task_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "task".into(),
        description: "Spawn a background subagent that runs asynchronously while you continue working. Returns immediately with a task_id — poll results with `get_task_output`, wait for multiple with `wait_tasks`, cancel with `kill_task`. Use `subagent_type` to choose: \"explore\" (read-only investigation) or \"delegate\" (write-capable implementation with verify-gated merge). Give ONE self-contained task with enough detail to complete standalone. Background subagents survive parent-turn cancellation — you can poll results later. The subagent cannot itself spawn subagents.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short description of the task (3-5 words)."
                },
                "prompt": {
                    "type": "string",
                    "description": "The full task prompt for the subagent to execute."
                },
                "subagent_type": {
                    "type": "string",
                    "enum": ["explore", "delegate"],
                    "description": "Type of subagent: \"explore\" (read-only) or \"delegate\" (write-capable with verify-gated merge). Default: \"explore\"."
                },
                "verify": {
                    "type": "string",
                    "description": "For delegate only: shell command that must pass for changes to be kept. If omitted, the session's verify command is used."
                }
            },
            "required": ["description", "prompt"]
        }),
    }
}

/// `get_task_output` — poll one or more background subagent tasks for output/status.
pub fn get_task_output_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "get_task_output".into(),
        description: "Poll one or more background subagent tasks for their current output and status. Returns immediately with current output and status (running/completed/failed/cancelled). For a single task, pass one task_id; for multiple, pass an array. Set a positive `timeout_ms` to wait up to that many milliseconds for completion (capped at ~10 min); omit or pass 0 for a non-blocking snapshot.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "task_ids": {
                    "description": "One task ID (string) or a list of task IDs (array of strings) to poll.",
                    "oneOf": [
                        { "type": "string" },
                        { "type": "array", "items": { "type": "string" } }
                    ]
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional max wait in milliseconds. 0 or omitted = non-blocking snapshot. Capped at ~10 min (600000ms). Default: 0."
                }
            },
            "required": ["task_ids"]
        }),
    }
}

/// `wait_tasks` — wait for multiple background subagent tasks to complete.
pub fn wait_tasks_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "wait_tasks".into(),
        description: "Wait for multiple background subagent tasks to complete. Prefer `get_task_output` with `task_ids` and a positive `timeout_ms`; this tool is kept for compatibility. Returns when all (mode=wait_all) or any (mode=wait_any) tasks complete, or the timeout expires.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of background task IDs to wait for."
                },
                "mode": {
                    "type": "string",
                    "enum": ["wait_all", "wait_any"],
                    "description": "wait_all (default) returns when all tasks complete; wait_any returns when any one completes."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional max wait in milliseconds. Default 30000, capped at ~10 min (600000ms)."
                }
            },
            "required": ["task_ids"]
        }),
    }
}

/// `kill_task` — cancel a running background subagent task.
pub fn kill_task_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "kill_task".into(),
        description: "Cancel a running background subagent task by its task_id. The subagent is terminated and its result (if any partial output was produced) becomes available via `get_task_output`. Idempotent — killing an already-completed task is a no-op.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task ID to cancel."
                }
            },
            "required": ["task_id"]
        }),
    }
}

/// `use_tool` — call an external MCP (Model Context Protocol) tool by name.
pub fn use_tool_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "use_tool".into(),
        description: "Call an external tool provided by a connected MCP (Model Context Protocol) server. Use `search_tool` first to discover available MCP tools and their parameters. Each MCP tool has its own parameter schema — pass the arguments as a JSON object in the `arguments` field.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "Name of the MCP server providing the tool."
                },
                "tool": {
                    "type": "string",
                    "description": "Name of the MCP tool to call."
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments object for the MCP tool, as defined by its schema.",
                    "additionalProperties": true
                }
            },
            "required": ["server", "tool"]
        }),
    }
}

/// `search_tool` — discover available MCP tools across connected servers.
pub fn search_tool_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "search_tool".into(),
        description: "Search for available external tools across connected MCP (Model Context Protocol) servers. Returns a list of tools with their names, descriptions, and parameter schemas. Use this to discover what MCP tools are available before calling them with `use_tool`.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional search query to filter tools by name or description. If omitted, lists all available tools."
                }
            }
        }),
    }
}

/// `memory_search` — search cross-session memory for relevant knowledge.
pub fn memory_search_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "memory_search".into(),
        description: "Search indexed cross-session memory for relevant knowledge — past decisions, coding facts, learned skills, and session summaries. Use this to recall context from previous sessions that isn't in the current conversation. Returns ranked chunks of memory text.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — what you want to recall from past sessions."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 5."
                }
            },
            "required": ["query"]
        }),
    }
}

/// `memory_get` — read a specific memory entry by its path.
pub fn memory_get_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "memory_get".into(),
        description: "Read a specific memory entry by its file path. Use after `memory_search` to retrieve the full content of a relevant memory chunk.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The memory file path to read."
                }
            },
            "required": ["path"]
        }),
    }
}

/// `skill` — invoke a named learned skill by name.
pub fn skill_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "skill".into(),
        description: "Invoke a named learned skill — a reusable procedure indexed from the project or user config. Skills encapsulate multi-step workflows (e.g. \"rust-workspace\", \"pytest-package\") and return their procedure text. Use this to apply a known skill to the current task rather than re-deriving the steps.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the skill to invoke."
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments to pass to the skill."
                }
            },
            "required": ["name"]
        }),
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical capability → side-effect class matrix for interactive tools.
    ///
    /// RSI `hi-tool-host::SideEffect` uses the same vocabulary (None / WorkspaceRead /
    /// WorkspaceWrite / Process / Network). Keep these mappings aligned when either
    /// catalog changes — see `hi-tool-host` tests for the host-side mirror.
    fn expected_side_effect_class(meta: &ToolMetadata) -> &'static str {
        if meta.filesystem_mutating {
            return "workspace_write";
        }
        match meta.capability {
            ToolCapability::Coordination => "none",
            ToolCapability::Repository | ToolCapability::Lsp => "workspace_read",
            ToolCapability::Mutation => "workspace_write",
            ToolCapability::Process | ToolCapability::Background | ToolCapability::Subagent => {
                "process"
            }
            ToolCapability::Mcp | ToolCapability::Memory | ToolCapability::Skill => {
                if meta.read_only { "network" } else { "process" }
            }
            ToolCapability::Web => {
                // web_download mutates via filesystem_mutating; search/fetch are network.
                if meta.read_only {
                    "network"
                } else {
                    "workspace_write"
                }
            }
        }
    }
    #[test]
    fn read_only_tools_are_classified() {
        assert!(is_read_only("read"));
        assert!(is_read_only("list"));
        assert!(is_read_only("grep"));
        assert!(is_read_only("diff"));
        assert!(is_read_only("glob"));
        // No filesystem side effects — safe to parallelize and offer in
        // read-only mode.
        assert!(is_read_only("update_plan"));
        assert!(is_read_only("record_decision"));
        assert!(is_read_only("bash_output"));
        // Mutating / effecting tools are not safe to run concurrently.
        assert!(!is_read_only("write"));
        assert!(!is_read_only("edit"));
        assert!(!is_read_only("multi_edit"));
        assert!(!is_read_only("apply_patch"));
        assert!(!is_read_only("bash"));
        assert!(!is_read_only("bash_kill"));
    }
    #[test]
    fn filesystem_mutating_tools_are_classified() {
        // Only tools that write to the working tree.
        assert!(is_filesystem_mutating("write"));
        assert!(is_filesystem_mutating("edit"));
        assert!(is_filesystem_mutating("multi_edit"));
        assert!(is_filesystem_mutating("apply_patch"));
        // Everything else — including non-read-only tools like bash — does not
        // directly mutate via the tool layer (bash runs alone; bash_kill stops
        // a process; update_plan/record_decision are in-memory only).
        assert!(!is_filesystem_mutating("bash"));
        assert!(!is_filesystem_mutating("bash_kill"));
        assert!(!is_filesystem_mutating("bash_output"));
        assert!(!is_filesystem_mutating("update_plan"));
        assert!(!is_filesystem_mutating("record_decision"));
        assert!(!is_filesystem_mutating("read"));
        assert!(!is_filesystem_mutating("diff"));
    }
    #[test]
    fn metadata_catalog_covers_every_schema_once() {
        let mut names = std::collections::BTreeSet::new();
        for metadata in TOOL_CATALOG {
            assert!(names.insert(metadata.name), "duplicate {}", metadata.name);
        }
        for spec in TOOL_SPECS.iter() {
            assert!(tool_metadata(&spec.name).is_some(), "missing {}", spec.name);
        }
        for spec in MINIMAL_TOOL_SPECS.iter() {
            assert!(
                tool_metadata(&spec.name).is_some_and(|metadata| metadata.minimal),
                "{} is not marked minimal",
                spec.name
            );
        }
        assert!(is_known_tool("explore"));
        assert!(is_known_tool("delegate"));
        assert!(!is_known_tool("hallucinated_tool"));
    }
    #[test]
    fn target_path_extracts_path_field() {
        assert_eq!(
            target_path("read", r#"{"path":"src/a.rs"}"#),
            Some("src/a.rs".into())
        );
        assert_eq!(
            target_path("write", r#"{"path":"b.rs","content":"x"}"#),
            Some("b.rs".into())
        );
        // list's path is optional → None when absent.
        assert_eq!(target_path("list", r#"{}"#), None);
        assert_eq!(target_path("list", r#"{"path":"sub"}"#), Some("sub".into()));
        // bash has no path → None (the safe-fallback case for dep inference).
        assert_eq!(target_path("bash", r#"{"command":"echo hi"}"#), None);
        // Malformed JSON → None (tolerant).
        assert_eq!(target_path("read", "not json"), None);
        // `read` with `paths`: a one-element array yields that path.
        assert_eq!(
            target_path("read", r#"{"paths":["src/a.rs"]}"#),
            Some("src/a.rs".into())
        );
        // A multi-element array has no single target → None.
        assert_eq!(
            target_path("read", r#"{"paths":["src/a.rs","src/b.rs"]}"#),
            None
        );
        // apply_patch: a single file directive's path is extracted.
        let patch =
            r#"{"patch":"*** Begin Patch\n*** Update File: src/a.rs\n-old\n+new\n*** End Patch"}"#;
        assert_eq!(target_path("apply_patch", patch), Some("src/a.rs".into()));
        let add_patch =
            r#"{"patch":"*** Begin Patch\n*** Add File: new.txt\nhello\n*** End Patch"}"#;
        assert_eq!(
            target_path("apply_patch", add_patch),
            Some("new.txt".into())
        );
        let delete_patch =
            r#"{"patch":"*** Begin Patch\n*** Delete File: old.txt\n*** End Patch"}"#;
        assert_eq!(
            target_path("apply_patch", delete_patch),
            Some("old.txt".into())
        );
        // Multi-file patches have no single target path. Returning None makes
        // dependency inference serialize later reads conservatively.
        let multi_patch = r#"{"patch":"*** Begin Patch\n*** Update File: src/a.rs\n-old\n+new\n*** Update File: src/b.rs\n-old\n+new\n*** End Patch"}"#;
        assert_eq!(target_path("apply_patch", multi_patch), None);
        // No file directives → None.
        assert_eq!(
            target_path(
                "apply_patch",
                r#"{"patch":"*** Begin Patch\n*** End Patch"}"#
            ),
            None
        );
    }
    #[test]
    fn minimal_tool_specs_is_a_lean_subset() {
        let full: Vec<&str> = TOOL_SPECS.iter().map(|s| s.name.as_str()).collect();
        let minimal: Vec<&str> = MINIMAL_TOOL_SPECS.iter().map(|s| s.name.as_str()).collect();
        assert!(minimal.len() < full.len());
        // Every minimal tool exists in the full set, in the same order.
        for name in &minimal {
            assert!(full.contains(name), "{name} missing from full specs");
        }
        // The essentials a small coding agent needs are present.
        for essential in [
            "read",
            "list",
            "grep",
            "repo_map",
            "find_symbol",
            "bash",
            "write",
            "edit",
        ] {
            assert!(
                minimal.contains(&essential),
                "{essential} missing from minimal"
            );
        }
    }
    #[test]
    fn capability_matrix_covers_every_catalog_entry() {
        assert!(!TOOL_CATALOG.is_empty());
        let mut names = std::collections::BTreeSet::new();
        for meta in TOOL_CATALOG {
            assert!(names.insert(meta.name), "duplicate tool {}", meta.name);
            let side = expected_side_effect_class(meta);
            // Invariants tying flags to side-effect class.
            match side {
                "none" => {
                    assert!(meta.read_only, "{} none must be read_only", meta.name);
                    assert!(!meta.filesystem_mutating);
                }
                "workspace_read" => {
                    assert!(
                        meta.read_only,
                        "{} workspace_read must be read_only",
                        meta.name
                    );
                    assert!(!meta.filesystem_mutating);
                }
                "workspace_write" => {
                    assert!(
                        meta.filesystem_mutating
                            || matches!(
                                meta.capability,
                                ToolCapability::Mutation | ToolCapability::Web
                            ),
                        "{} workspace_write should mutate fs or be Mutation/Web",
                        meta.name
                    );
                    assert!(
                        !meta.read_only,
                        "{} workspace_write must not be read_only",
                        meta.name
                    );
                }
                "process" => {
                    assert!(
                        matches!(
                            meta.capability,
                            ToolCapability::Process
                                | ToolCapability::Background
                                | ToolCapability::Subagent
                                | ToolCapability::Mcp
                        ),
                        "{} process class capability",
                        meta.name
                    );
                }
                "network" => {
                    assert!(
                        matches!(
                            meta.capability,
                            ToolCapability::Web
                                | ToolCapability::Mcp
                                | ToolCapability::Memory
                                | ToolCapability::Skill
                        ),
                        "{} network class capability",
                        meta.name
                    );
                    assert!(meta.read_only);
                }
                other => panic!("unknown side effect class {other}"),
            }
            // Classifier helpers stay consistent with catalog flags.
            assert_eq!(is_read_only(meta.name), meta.read_only);
            assert_eq!(is_filesystem_mutating(meta.name), meta.filesystem_mutating);
            assert_eq!(
                is_coordination(meta.name),
                meta.capability == ToolCapability::Coordination
            );
            assert!(is_known_tool(meta.name));
        }
    }
    #[test]
    fn capability_matrix_known_tool_side_effects() {
        // Explicit pins so a casual catalog edit fails loudly.
        let pins = [
            ("update_plan", "none"),
            ("record_decision", "none"),
            // Records goal bookkeeping only; touches no file and runs nothing.
            ("block_step", "none"),
            ("read", "workspace_read"),
            ("list", "workspace_read"),
            ("grep", "workspace_read"),
            ("glob", "workspace_read"),
            ("repo_map", "workspace_read"),
            ("find_symbol", "workspace_read"),
            ("diff", "workspace_read"),
            ("diagnostics", "workspace_read"),
            ("definition", "workspace_read"),
            ("references", "workspace_read"),
            ("hover", "workspace_read"),
            ("write", "workspace_write"),
            ("edit", "workspace_write"),
            ("multi_edit", "workspace_write"),
            ("apply_patch", "workspace_write"),
            ("web_download", "workspace_write"),
            ("bash", "process"),
            ("bash_output", "process"),
            ("bash_kill", "process"),
            ("explore", "process"),
            ("delegate", "process"),
            ("task", "process"),
            ("get_task_output", "process"),
            ("wait_tasks", "process"),
            ("kill_task", "process"),
            ("web_search", "network"),
            ("web_fetch", "network"),
            ("search_tool", "network"),
            ("use_tool", "process"),
            ("memory_search", "network"),
            ("memory_get", "network"),
            ("skill", "network"),
        ];
        for (name, want) in pins {
            let meta = tool_metadata(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(
                expected_side_effect_class(meta),
                want,
                "{name} side-effect class drifted"
            );
        }
        // Every catalog entry is pinned (no silent additions).
        let pinned: std::collections::BTreeSet<_> = pins.iter().map(|(n, _)| *n).collect();
        for meta in TOOL_CATALOG {
            assert!(
                pinned.contains(meta.name),
                "add an explicit side-effect pin for new tool `{}`",
                meta.name
            );
        }
    }
}
