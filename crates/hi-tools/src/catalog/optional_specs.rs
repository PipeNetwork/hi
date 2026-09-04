//! Injected and optional tool specifications.

use hi_ai::ToolSpec;
use serde_json::json;

/// The provider-facing envelope for a restricted Rhai program. It is kept out
/// of the global specs so providers without native tool calling see no schema.
pub fn run_program_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "run_program".into(),
        description: "Execute a bounded Rhai program. The final expression is returned. Use `tool(name, #{...})` for existing tools and `parallel([#{name: \"read\", args: #{path: \"src/lib.rs\"}}])` for independent calls. No filesystem, process, network, imports, dynamic evaluation, time, sleep, or exit functions are available; only approved host tools may run.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "maxLength": 262144,
                    "description": "Rhai source whose final expression is the program result."
                }
            },
            "required": ["source"],
            "additionalProperties": false
        }),
    }
}

/// The `explore` read-only subagent tool. Deliberately kept OUT of [`super::TOOL_SPECS`]
/// and out of [`super::is_read_only`]: it's only advertised when the agent explicitly
/// injects it (for a capable parent via `explore_subagents`), and because it's not
/// read-only it never survives into a `ReadOnly` child's tool set — so a subagent
/// cannot spawn another (depth is capped at 1 structurally).
pub fn explore_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "explore".into(),
        description: "Delegate a focused, READ-ONLY investigation to a subagent that runs in its own fresh context and returns just a concise answer. Use it to keep your own context clean when a question needs reading or searching across many files. For parallel investigations, split work into independent scopes and name exact target files or directories when known. The subagent can only read/list/grep/glob and inspect code (no edits, no shell, no spawning). Give it ONE self-contained task with enough detail to answer standalone. Prefer it over reading many files yourself when you only need the conclusion; don't use it for trivial single-file lookups or anything that must change files.".into(),
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
/// OUT of [`super::TOOL_SPECS`] and [`super::is_read_only`], and is only injected for a top-level
/// agent (via `write_subagents`) — never for a subagent, so it can't recurse.
pub fn delegate_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "delegate".into(),
        description: "Delegate a self-contained IMPLEMENTATION subtask to a subagent that runs in its own fresh context, can edit files and run commands, and verifies its own work. Its changes are merged back ONLY if verification passes. For parallel delegates, decompose work into independent, non-overlapping file or directory scopes and name the exact paths each task owns; unknown or overlapping scopes must not be parallelized. Give ONE standalone task with clear success criteria. Prefer doing small edits yourself; use this for a substantial, independently-verifiable subtask. The subagent cannot itself delegate or explore.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A single, self-contained implementation subtask, including what 'done' looks like."
                },
                "scope": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional authoritative workspace-relative files or directories owned by this delegate. Use distinct, non-overlapping scopes for delegates that should run in parallel. If omitted, paths are conservatively inferred from the task text."
                },
                "verify": {
                    "type": "string",
                    "description": "Optional shell command that must pass for the subagent's changes to be kept (e.g. `cargo test foo`). If omitted, the session's verify command is used."
                },
                "kind": {
                    "type": "string",
                    "enum": ["author", "edit"],
                    "description": "Task shape. \"edit\" = a mechanical, precisely-specified change (rename, small targeted fix, apply a described diff, config tweak) — may run on a faster editor model when the session configures one. \"author\" (default) = writing new code or any open-ended change."
                }
            },
            "required": ["task"]
        }),
    }
}

/// The `task` tool — spawns a background subagent that runs asynchronously while
/// the parent continues working. Returns immediately with a task handle; poll
/// results with `get_task_output` or `wait_tasks`, cancel with `kill_task`.
/// Built-in kinds match grok-build: `explore`, `plan`, `general-purpose`.
/// Like `explore`/`delegate`, kept OUT of `TOOL_SPECS` and injected only for a
/// top-level agent.
pub fn task_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "task".into(),
        description: "Spawn a background subagent that runs asynchronously while you continue working. Returns immediately with a task_id — poll results with `get_task_output`, wait for multiple with `wait_tasks`, cancel with `kill_task`. Use `subagent_type` to choose \"explore\" (fast read-only investigation), \"plan\" (read-only architecture/implementation planning), or \"general-purpose\" (write-capable detached candidate). General-purpose children never edit the live workspace: the parent verifies their exact base and applies them transactionally at a safe turn boundary. Give ONE self-contained task with enough detail to complete standalone. Background subagents survive parent-turn cancellation. The subagent cannot itself spawn subagents.".into(),
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
                    "enum": ["explore", "plan", "general-purpose"],
                    "description": "Built-in subagent type: \"explore\" (read-only investigation), \"plan\" (read-only planning), or \"general-purpose\" (verified detached write candidate). Default: \"explore\"."
                },
                "depends_on": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional task IDs that must complete successfully before this task is ready."
                },
                "cost": {
                    "type": "string",
                    "enum": ["tiny", "normal", "large"],
                    "description": "Optional work-size hint. Tiny edits should usually remain in the parent unless compatible work can be coalesced."
                },
                "verify": {
                    "type": "string",
                    "description": "Optional verification command required before a general-purpose candidate can become ready to merge."
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

/// Admission (ADR 002): asking in assistant text cannot pause the turn, so the
/// human path fails the structure/control-plane gate. Inject-only for
/// interactive parent sessions — not in [`super::TOOL_SPECS`], [`super::MINIMAL_TOOL_SPECS`],
/// or [`super::PROTECTED_TOOLS`]. Side-effect class: none (Coordination, read-only).
pub fn ask_user_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "ask_user".into(),
        description: "Pause and ask the user a product or design question that tools cannot resolve. Call only when you are blocked on a real choice (API shape, UX copy, which approach to take) — not instead of keep-working on a known next coding step, and not for confirmations the user already answered. Product/design forks only; never instead of the next coding step. Pass `question` and optional `options`. The tool result is the user's answer. If the frontend cannot pause, pick the best option yourself and continue.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask. Be specific about the decision and why tools cannot resolve it."
                },
                "options": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional short choices the user can pick from. They may still type a custom answer."
                }
            },
            "required": ["question"]
        }),
    }
}

/// Admission (ADR 002): Structure — `/compact` summarizes (wrong when the
/// window is poisoned) and `/window` is user-only. Inject-only, occupancy-gated,
/// sticky once advertised. Side-effect class: none (Coordination, read-only).
pub fn new_context_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "new_context".into(),
        description: "Start a new context window without summarizing. Drops conversation history and keeps the current task, goal, and decisions. Call only when this window is no longer useful (failed approach, topic change, contradictory evidence) and occupancy is already high — not to save a little room (use smaller reads) and not instead of `/compact` when you still need a handoff brief. Empty arguments. At most once per turn.".into(),
        parameters: json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
    }
}

/// Admission (ADR 002): Structure + Reliability over `web_search`+`web_fetch`.
/// Inject-only — not in [`super::TOOL_SPECS`], [`super::MINIMAL_TOOL_SPECS`], or
/// [`super::PROTECTED_TOOLS`]. Side-effect class: network.
pub fn research_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "research".into(),
        description: "Research the live web via Pipe POST /v1/research: multiple search terms, page scrapes, embeddings + rerank, top snippets. Prefer this over guessing URLs with web_fetch when the user asked to research on the web. Keep web_search for cheap lookups and web_fetch when the user already gave a URL. The result includes research_id and page_id handles — follow up with research_read for a useful page.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research question or search topic."
                }
            },
            "required": ["query"]
        }),
    }
}

/// Follow-up page read for a `research` session. Inject-only, same admission.
pub fn research_read_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "research_read".into(),
        description: "Read cached markdown for a page from a prior `research` call. Pass research_id and page_id from that result. 404 if the session expired.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "research_id": { "type": "string", "description": "research_id from the research tool result." },
                "page_id": { "type": "string", "description": "page_id handle from a snippet or pages list." }
            },
            "required": ["research_id", "page_id"]
        }),
    }
}

/// Admission (ADR 002): Safety + Reliability over `bash`+curl for page/login/UI
/// work. Inject/feature-gated — not in [`super::TOOL_SPECS`], [`super::MINIMAL_TOOL_SPECS`],
/// or [`super::PROTECTED_TOOLS`]. Side-effect class: network. `browser_click` stays
/// Reject; this is one `browser_exec` mini-language.
pub fn browser_exec_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "browser_exec".into(),
        description: "Drive a real browser with a short script (goto, click, type, screenshot, ax, wait, eval, scroll). Prefer this over `bash` curl/wget when the user asked to open a page, debug a login form, or inspect live UI. On by default; set `[browser] enabled = false` in hi.toml to hide it. Do not use for ordinary coding, file edits, or fetching a known URL (`web_fetch`). Cloud metadata and link-local hosts are always blocked.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "One command per line: goto <url>, click <index|x y>, type [index] <text>, screenshot, ax, wait <ms>, eval <js>, scroll <dx> <dy>."
                },
                "mode": {
                    "type": "string",
                    "enum": ["headless", "dedicated"],
                    "description": "headless (default) or dedicated; both launch a fresh owned Chrome so the network guard is installed before navigation."
                }
            },
            "required": ["script"]
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

/// `memory_search` — search markdown session memory. Inject-only (not in [`super::TOOL_SPECS`]).
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

/// `memory_get` — read a markdown memory bullet. Inject-only.
pub fn memory_get_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "memory_get".into(),
        description: "Read a specific memory entry by id (`#12` or `project:#12`) or file path. Use after `memory_search`.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Memory ref: `#12`, `project:#12`, `global:#12`, or a markdown file path."
                }
            },
            "required": ["path"]
        }),
    }
}

/// `memory_update` — replace a markdown memory bullet by stable id. Inject-only.
pub fn memory_update_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "memory_update".into(),
        description: "Replace a durable markdown memory bullet by its `[#n]` id. Refused when memory is disabled (`--no-memory`). After a save the UI offers `/undo-memory`.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The bullet id, e.g. 12 for `[#12]`."
                },
                "text": {
                    "type": "string",
                    "description": "Replacement bullet text (without the `- [#n]` prefix)."
                }
            },
            "required": ["id", "text"]
        }),
    }
}

/// `memory_forget` — drop a markdown memory bullet by stable id. Inject-only.
pub fn memory_forget_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "memory_forget".into(),
        description: "Remove a durable markdown memory bullet by its `[#n]` id. Refused when memory is disabled (`--no-memory`). After a save the UI offers `/undo-memory`.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The bullet id, e.g. 12 for `[#12]`."
                }
            },
            "required": ["id"]
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
