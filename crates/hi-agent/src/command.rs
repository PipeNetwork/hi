//! Slash-command parsing, shared by every frontend.

/// A recognized in-session command. Frontends decide how to act on each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Help,
    /// Reset the conversation, keeping only the system prompt.
    Clear,
    /// Set the model for subsequent turns (empty = report current).
    Model(String),
    /// Run exactly one turn through the conservative MoA virtual route.
    Moa(String),
    /// Use a provider/profile for subsequent turns (empty = report current).
    /// Named profiles are resolved from the config; the model can be set later
    /// with `/model` when the profile does not configure one.
    ///
    /// Subcommands: `add` (create a new profile interactively), `edit [name]`
    /// (edit an existing profile). The frontend parses these from the arg.
    Provider(String),
    /// Show current session/runtime status.
    Status,
    /// Toggle or query the LSP subsystem. Arg: `on`, `off`, or empty (status).
    Lsp(String),
    /// Toggle or query the write-capable `delegate` subagent. Arg: `on`, `off`,
    /// or empty (status).
    Delegate(String),
    /// Expanded read-only slash prompt macro that should run as a model turn.
    Prompt(String),
    /// Write a debug/event log for the current session.
    Log,
    /// Show, set, or clear the verify command turns iterate against. Empty =
    /// show; `off`/`none`/`clear` = disable; anything else = set.
    Verify(String),
    /// Show what's changed in the working tree (git diff).
    Diff,
    /// Copy the last assistant response, or `all` for the transcript.
    Copy(String),
    /// Show, set, or clear the current session goal.
    Goal(String),
    /// Show a context-occupancy breakdown: system prompt, per-turn token
    /// estimates, and what compaction would keep/elide.
    Context,
    /// Explore the repo and write a project-context file (runs as a turn).
    Init,
    /// Learn a reusable workflow and write one local SKILL.md (runs as a turn).
    Learn(String),
    /// List discovered project/global learned skills.
    Skills,
    /// Use a learned skill by name as the next model turn.
    Skill(String),
    /// Reclaim context. Empty arg = configured strategy; `full`/`hybrid`/`elide`
    /// select one explicitly.
    Compact(String),
    /// Re-run the last user message (after truncating its previous attempt).
    Retry,
    /// Load the last user prompt into the input line for editing before
    /// resending. Handled by the frontend (it manipulates the input line).
    Edit,
    /// Revert the file changes the last turn made (from its git checkpoint).
    Undo,
    /// Stage all working-tree changes and commit them with an auto-generated
    /// message summarizing the changed files (the `/commit` command).
    Commit,
    /// Print the version and exit.
    Version,
    /// Export the conversation to a file.
    Export(String),
    /// Inspect the configured MCP endpoint: server info, tools, model count.
    Mcp,
    /// Discover/list/download Hugging Face Hub model artifacts.
    Hf(String),
    /// Open the fleet dashboard: dispatch, monitor, and steer multiple
    /// concurrent agent sessions from one screen (TUI only). Arg: empty opens
    /// the dashboard; `status` lists this project's resumable fleet sessions.
    Dashboard(String),
    /// Recurring agent turns on a cadence (TUI only): `<interval> <prompt>`
    /// creates, empty/`list` lists, `cancel <id>` stops one.
    Loop(String),
    /// Full-screen live dashboard of all active loops (TUI only).
    Watch,
    /// Show the activity digest: what loops have noticed, grouped by loop, with
    /// what's new since you last looked (TUI only).
    Digest,
    Quit,
    /// A `/word` that isn't recognized.
    Unknown(String),
    /// A removed command, retained as a redirect. Carries a hint shown verbatim.
    Removed(String),
}

/// Parse a line as a command. Returns `None` for ordinary input (anything not
/// starting with `/`).
pub fn parse(line: &str) -> Option<Command> {
    let line = line.trim();
    let rest = line.strip_prefix('/')?;
    let mut parts = rest.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim().to_string();
    Some(match name {
        "help" | "h" | "?" => Command::Help,
        "clear" | "new" => Command::Clear,
        "model" | "m" => Command::Model(arg),
        "moa" => Command::Moa(arg),
        "provider" | "prov" => Command::Provider(arg),
        "usage" | "cost" => Command::Removed("usage — removed; use /status".into()),
        "review" => Command::Prompt(read_only_macro_prompt("review", &arg)),
        "security" | "audit" => Command::Prompt(read_only_macro_prompt("security", &arg)),
        "roadmap" => Command::Prompt(read_only_macro_prompt("roadmap", &arg)),
        "gaps" => Command::Prompt(read_only_macro_prompt("gaps", &arg)),
        "build" => Command::Prompt(build_macro_prompt(&arg)),
        "status" | "st" if arg.is_empty() => Command::Status,
        "status" | "st" => Command::Prompt(read_only_macro_prompt("status", &arg)),
        "log" | "debug" => Command::Log,
        "verify" | "test" => Command::Verify(arg),
        "diff" | "changes" => Command::Diff,
        "copy" | "cp" => Command::Copy(arg),
        "goal" => Command::Goal(arg),
        "context" | "ctx" => Command::Context,
        "init" => Command::Init,
        "learn" => Command::Learn(arg),
        "skills" => Command::Skills,
        "skill" => Command::Skill(arg),
        "compact" => Command::Compact(arg),
        "retry" | "redo" => Command::Retry,
        "edit" => Command::Edit,
        "undo" | "revert" => Command::Undo,
        "commit" => Command::Commit,
        "version" | "ver" | "v" => Command::Version,
        "export" => Command::Export(arg),
        "mcp" => Command::Mcp,
        "hf" | "hd" | "huggingface" => Command::Hf(arg),
        "lsp" => Command::Lsp(arg),
        "delegate" | "delegates" => Command::Delegate(arg),
        "dashboard" | "fleet" => Command::Dashboard(arg),
        "loop" | "loops" => Command::Loop(arg),
        "watch" => Command::Watch,
        "digest" | "activity" => Command::Digest,
        "exit" | "quit" | "q" => Command::Quit,
        other => Command::Unknown(other.to_string()),
    })
}

/// Expand a read-only prompt slash macro, or return `None` for non-macros.
pub fn expand_prompt_macro(line: &str) -> Option<String> {
    match parse(line)? {
        Command::Prompt(prompt) => Some(prompt),
        _ => None,
    }
}

/// Whether a `/goal` argument is an objective to plan/decompose, versus a control
/// subcommand (empty, `clear`/`off`/`none`, `pause`, `resume`, or `limit …`).
/// Frontends use this to route only real objectives to the planner.
pub fn goal_arg_is_objective(arg: &str) -> bool {
    let a = arg.trim();
    !(a.is_empty()
        || matches!(a, "clear" | "off" | "none" | "pause" | "resume")
        || a == "limit"
        || a.starts_with("limit "))
}

/// Parse a loop interval like `60s`, `90s`, `30m`, `2h`, `1d` into seconds.
/// Bounds: 60 seconds to 7 days. Bare numbers are seconds.
pub fn parse_loop_interval(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (num, unit) = match s.chars().last() {
        Some(c) if c.is_ascii_digit() => (s, "s"),
        Some('s') => (&s[..s.len() - 1], "s"),
        Some('m') => (&s[..s.len() - 1], "m"),
        Some('h') => (&s[..s.len() - 1], "h"),
        Some('d') => (&s[..s.len() - 1], "d"),
        _ => return None,
    };
    let n: u64 = num.parse().ok()?;
    let secs = n.checked_mul(match unit {
        "m" => 60,
        "h" => 3600,
        "d" => 86_400,
        _ => 1,
    })?;
    (60..=7 * 86_400).contains(&secs).then_some(secs)
}

/// Parse a token count like `500k`, `1.5m`, `250000` into a number. Bare
/// numbers are exact; `k`/`m` are ×1_000 / ×1_000_000 (decimals allowed).
pub fn parse_token_count(s: &str) -> Option<u64> {
    let s = s.trim().to_lowercase();
    if s.is_empty() {
        return None;
    }
    let (num, mult): (&str, u64) = match s.chars().last()? {
        'k' => (&s[..s.len() - 1], 1_000),
        'm' => (&s[..s.len() - 1], 1_000_000),
        c if c.is_ascii_digit() => (s.as_str(), 1),
        _ => return None,
    };
    let n: f64 = num.trim().parse().ok()?;
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    Some((n * mult as f64).round() as u64)
}

/// The parsed form of a `/loop` argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopArg {
    /// Empty or `list` — show active loops.
    List,
    /// `cancel <id>`.
    Cancel(u64),
    /// `pause <id>` — hold a loop (stops firing; stays resumable).
    Pause(u64),
    /// `resume <id>` — resume a paused loop.
    Resume(u64),
    /// `budget <id> <count|off>` — set/clear a token spend cap (auto-pauses).
    Budget { id: u64, tokens: Option<u64> },
    /// `on <id> <cmd|off>` — set/clear a shell command run when a firing is loud.
    Trigger { id: u64, cmd: Option<String> },
    /// `fix <id> <on|pr|off>` — enable/disable auto-fix (dispatch a verified fix
    /// on a loud change); `pr` lands it as a PR instead of a working-tree merge.
    Fix { id: u64, on: bool, pr: bool },
    /// `window <id> <H-H [weekdays]|off>` — only fire within a local-time window.
    /// `Some((start_hour, end_hour, weekdays_only))`, or `None` to clear.
    Window {
        id: u64,
        window: Option<(u8, u8, bool)>,
    },
    /// `cost` — a token-spend breakdown across loops.
    Cost,
    /// `<interval> <prompt>` — create a loop firing `prompt` every `secs`.
    Create { secs: u64, prompt: String },
    /// Anything unparseable (bad interval / missing prompt / bad id).
    Invalid(String),
}

/// Parse a bare loop id (tolerating a leading `#`).
fn parse_loop_id(s: &str) -> Result<u64, String> {
    let s = s.trim().trim_start_matches('#');
    s.parse()
        .map_err(|_| format!("bad loop id '{s}' — /loop list shows ids"))
}

/// Split a `/loop` argument into its subcommand form.
/// Parse a fire-window spec: `H-H` (hours 0..24) with an optional `weekdays`
/// (or `mon-fri`) token → `(start_hour, end_hour, weekdays_only)`.
pub fn parse_loop_window(s: &str) -> Option<(u8, u8, bool)> {
    let s = s.trim();
    let mut parts = s.split_whitespace();
    let range = parts.next()?;
    let (a, b) = range.split_once('-')?;
    let start: u8 = a.trim().parse().ok()?;
    let end: u8 = b.trim().parse().ok()?;
    if start > 23 || end > 24 || start == end {
        return None;
    }
    let weekdays = match parts.next() {
        None => false,
        Some("weekdays" | "mon-fri" | "weekday") => true,
        Some(_) => return None,
    };
    Some((start, end, weekdays))
}

pub fn parse_loop_arg(arg: &str) -> LoopArg {
    let a = arg.trim();
    if a.is_empty() || a == "list" || a == "ls" || a == "status" {
        return LoopArg::List;
    }
    if matches!(a, "cost" | "spend") {
        return LoopArg::Cost;
    }
    if let Some(rest) = a.strip_prefix("window") {
        let rest = rest.trim();
        let Some((id_str, spec)) = rest.split_once(char::is_whitespace) else {
            return LoopArg::Invalid(
                "usage: /loop window <id> <9-17 [weekdays]|off>  (local time)".into(),
            );
        };
        let id = match parse_loop_id(id_str) {
            Ok(id) => id,
            Err(msg) => return LoopArg::Invalid(msg),
        };
        let spec = spec.trim();
        if matches!(spec, "off" | "none" | "clear" | "always") {
            return LoopArg::Window { id, window: None };
        }
        return match parse_loop_window(spec) {
            Some(w) => LoopArg::Window {
                id,
                window: Some(w),
            },
            None => LoopArg::Invalid(format!(
                "bad window '{spec}' — use e.g. 9-17, or 9-17 weekdays, or off"
            )),
        };
    }
    if let Some(rest) = a.strip_prefix("cancel") {
        return match parse_loop_id(rest) {
            Ok(id) => LoopArg::Cancel(id),
            Err(msg) => LoopArg::Invalid(msg),
        };
    }
    if let Some(rest) = a.strip_prefix("pause") {
        return match parse_loop_id(rest) {
            Ok(id) => LoopArg::Pause(id),
            Err(msg) => LoopArg::Invalid(msg),
        };
    }
    if let Some(rest) = a.strip_prefix("resume") {
        return match parse_loop_id(rest) {
            Ok(id) => LoopArg::Resume(id),
            Err(msg) => LoopArg::Invalid(msg),
        };
    }
    if let Some(rest) = a.strip_prefix("budget") {
        let rest = rest.trim();
        let Some((id_str, amount)) = rest.split_once(char::is_whitespace) else {
            return LoopArg::Invalid(
                "usage: /loop budget <id> <count|off> — e.g. /loop budget 3 500k".into(),
            );
        };
        let id = match parse_loop_id(id_str) {
            Ok(id) => id,
            Err(msg) => return LoopArg::Invalid(msg),
        };
        let amount = amount.trim();
        if matches!(amount, "off" | "none" | "clear" | "0") {
            return LoopArg::Budget { id, tokens: None };
        }
        return match parse_token_count(amount) {
            Some(tokens) => LoopArg::Budget {
                id,
                tokens: Some(tokens),
            },
            None => LoopArg::Invalid(format!(
                "bad token count '{amount}' — use e.g. 500k, 1.5m, or off"
            )),
        };
    }
    if a == "on" || a.starts_with("on ") {
        let rest = a[2..].trim();
        let Some((id_str, cmd)) = rest.split_once(char::is_whitespace) else {
            return LoopArg::Invalid(
                "usage: /loop on <id> <command>  (runs when a firing is loud; `off` clears)".into(),
            );
        };
        let id = match parse_loop_id(id_str) {
            Ok(id) => id,
            Err(msg) => return LoopArg::Invalid(msg),
        };
        let cmd = cmd.trim();
        let cmd = if matches!(cmd, "off" | "none" | "clear" | "") {
            None
        } else {
            Some(cmd.to_string())
        };
        return LoopArg::Trigger { id, cmd };
    }
    if let Some(rest) = a.strip_prefix("fix ") {
        let Some((id_str, state)) = rest.trim().split_once(char::is_whitespace) else {
            return LoopArg::Invalid(
                "usage: /loop fix <id> on|off  (dispatch a verified auto-fix on a loud change)"
                    .into(),
            );
        };
        let id = match parse_loop_id(id_str) {
            Ok(id) => id,
            Err(msg) => return LoopArg::Invalid(msg),
        };
        return match state.trim() {
            "on" | "yes" | "true" => LoopArg::Fix {
                id,
                on: true,
                pr: false,
            },
            "pr" => LoopArg::Fix {
                id,
                on: true,
                pr: true,
            },
            "off" | "no" | "false" => LoopArg::Fix {
                id,
                on: false,
                pr: false,
            },
            other => LoopArg::Invalid(format!("say on, pr, or off, not '{other}'")),
        };
    }
    let Some((head, prompt)) = a.split_once(char::is_whitespace) else {
        return LoopArg::Invalid(
            "usage: /loop <interval> <prompt> — e.g. /loop 30m check whether CI is green".into(),
        );
    };
    match parse_loop_interval(head) {
        Some(secs) if !prompt.trim().is_empty() => LoopArg::Create {
            secs,
            prompt: prompt.trim().to_string(),
        },
        Some(_) => LoopArg::Invalid("missing prompt after the interval".into()),
        None => LoopArg::Invalid(format!(
            "bad interval '{head}' — use 60s..7d (e.g. 90s, 30m, 2h, 1d)"
        )),
    }
}

/// The parsed value of a `/goal limit …` subcommand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoalLimitArg {
    /// `/goal limit <n>` — cap the plan at `n` sub-goals.
    Set(usize),
    /// `/goal limit off|none|clear|0` — remove the cap (the default: grow freely).
    Unlimited,
    /// `/goal limit` — report the current limit.
    Show,
    /// `/goal limit <garbage>` — the value didn't parse.
    Invalid(String),
}

/// Parse the argument of a `/goal limit …` subcommand. Returns `None` when `arg`
/// isn't a `limit` subcommand at all (so the caller handles other `/goal` forms).
pub fn parse_goal_limit(arg: &str) -> Option<GoalLimitArg> {
    let a = arg.trim();
    let rest = if a == "limit" {
        ""
    } else {
        a.strip_prefix("limit ")?.trim()
    };
    Some(match rest {
        "" => GoalLimitArg::Show,
        "off" | "none" | "clear" => GoalLimitArg::Unlimited,
        value => match value.parse::<usize>() {
            Ok(0) => GoalLimitArg::Unlimited,
            Ok(n) => GoalLimitArg::Set(n),
            Err(_) => GoalLimitArg::Invalid(value.to_string()),
        },
    })
}

fn read_only_macro_prompt(kind: &str, topic: &str) -> String {
    let topic = topic.trim();
    let topic = if topic.is_empty() {
        "the codebase"
    } else {
        topic
    };
    let recipe = match kind {
        "security" => {
            "Search for unsafe, unwrap, expect, panic!, command execution, filesystem/env access, and secret/token/auth patterns, then read the top matching files."
        }
        "status" => {
            "Inspect git status/diff summary, workspace manifests, README/docs when present, main crate or module entrypoints, and tests."
        }
        "roadmap" => {
            "Inspect workspace manifests, owning modules, tests, and TODO/FIXME or missing-coverage search results before naming build-next work."
        }
        "gaps" => {
            "Inspect workspace manifests, owning modules, tests, and TODO/FIXME or missing-coverage search results before naming gaps."
        }
        _ => "Inspect relevant files or targeted search results before giving findings.",
    };
    format!(
        "Read-only {kind} request for: {topic}\n\nDo not write, edit, apply patches, run mutating shell commands, or change files. Use read-only inspection before the final answer. {recipe}\n\nIf only a directory listing is available, keep inspecting or explicitly say the evidence is insufficient instead of making file-specific findings."
    )
}

fn build_macro_prompt(topic: &str) -> String {
    let topic = topic.trim();
    let topic = if topic.is_empty() {
        "the requested tool"
    } else {
        topic
    };
    format!(
        "Build {topic}.\n\nImplementation requirements:\n- Inspect the workspace before choosing files or stack.\n- Choose the local stack implied by existing manifests and entrypoints; if no stack is clear and this is a TUI, create a Rust binary in the current directory using Ratatui and Crossterm.\n- In an empty Rust target directory, prefer `cargo init --bin .` before editing so the manifest has a valid target from the start.\n- Edit or create the required files; do not stop at a plan, explanation, or scaffold.\n- Prefer a compact working vertical slice and small valid tool calls over one huge all-at-once source write.\n- Run an appropriate noninteractive validation command after the last file change.\n- Finish with a concise recap naming changed files and validation commands."
    )
}

/// One user-facing slash command — the single source of truth for `/help` and
/// the interactive completion menu, so they can't drift from each other.
pub struct CommandSpec {
    /// Canonical name without the leading slash (what completion inserts).
    pub name: &'static str,
    /// Argument hint, e.g. `[id]`; empty when the command takes no arguments.
    pub args: &'static str,
    /// One-line description.
    pub help: &'static str,
    /// Enumerable values the argument can take, each with a one-line hint, for
    /// the completion menu (e.g. `/compact ` → hybrid/full/elide). Empty when the
    /// argument is freeform (`/model <id>`, `/goal <text>`) or absent.
    pub arg_values: &'static [(&'static str, &'static str)],
}

impl CommandSpec {
    /// Whether the command accepts arguments (so completion leaves a trailing
    /// space for the user to type them, rather than submitting immediately).
    pub fn takes_args(&self) -> bool {
        !self.args.is_empty()
    }
}

/// Every slash command, in display order. Each `name` must be parseable by
/// [`parse`] (guarded by a test).
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "help",
        args: "",
        help: "show this help",
        arg_values: &[],
    },
    CommandSpec {
        name: "model",
        args: "[id]",
        help: "show or set the model (no id opens the selector)",
        arg_values: &[],
    },
    CommandSpec {
        name: "moa",
        args: "<prompt>",
        help: "run one prompt through moa/conservative, then restore the current model",
        arg_values: &[],
    },
    CommandSpec {
        name: "provider",
        args: "[name|add|edit|remove]",
        help: "use a profile, or add/edit/remove a profile (no arg lists all)",
        arg_values: &[],
    },
    CommandSpec {
        name: "verify",
        args: "[cmd|off]",
        help: "show/set/clear the test command turns iterate against",
        arg_values: &[("off", "disable the verify command")],
    },
    CommandSpec {
        name: "review",
        args: "[topic]",
        help: "run a read-only code review with file inspection",
        arg_values: &[],
    },
    CommandSpec {
        name: "security",
        args: "[topic]",
        help: "run a read-only security review with targeted search",
        arg_values: &[],
    },
    CommandSpec {
        name: "audit",
        args: "[topic]",
        help: "run a read-only security audit with targeted search",
        arg_values: &[],
    },
    CommandSpec {
        name: "roadmap",
        args: "[topic]",
        help: "discuss build-next roadmap after inspection",
        arg_values: &[],
    },
    CommandSpec {
        name: "gaps",
        args: "[topic]",
        help: "discuss missing gaps after inspection",
        arg_values: &[],
    },
    CommandSpec {
        name: "build",
        args: "[thing]",
        help: "build a tool/app end-to-end with edits and validation",
        arg_values: &[],
    },
    CommandSpec {
        name: "diff",
        args: "",
        help: "show what files have changed (git diff)",
        arg_values: &[],
    },
    CommandSpec {
        name: "copy",
        args: "[all]",
        help: "copy the last response (or transcript) to the clipboard",
        arg_values: &[("all", "copy the whole transcript, not just the last reply")],
    },
    CommandSpec {
        name: "goal",
        args: "[text|pause|resume|limit N|clear]",
        help: "set a goal (planner-decomposed, grows as it works), or pause/resume/limit/clear it",
        arg_values: &[
            (
                "pause",
                "pause the goal — hold progress, stop steering turns",
            ),
            ("resume", "resume a paused goal"),
            (
                "limit",
                "cap plan growth: /goal limit <n>, or 'limit off' for none",
            ),
            ("clear", "clear the current goal"),
        ],
    },
    CommandSpec {
        name: "context",
        args: "",
        help: "show context-occupancy breakdown and compaction preview",
        arg_values: &[],
    },
    CommandSpec {
        name: "init",
        args: "",
        help: "scan the repo and write an HI.md project guide",
        arg_values: &[],
    },
    CommandSpec {
        name: "learn",
        args: "[request]",
        help: "write one reusable local SKILL.md from a workflow",
        arg_values: &[],
    },
    CommandSpec {
        name: "skills",
        args: "",
        help: "list learned project/global skills",
        arg_values: &[],
    },
    CommandSpec {
        name: "skill",
        args: "<name>",
        help: "inject a learned skill for the next turn",
        arg_values: &[],
    },
    CommandSpec {
        name: "compact",
        args: "[kind]",
        help: "reclaim context (kind: hybrid, full, or elide)",
        arg_values: &[
            (
                "hybrid",
                "summarize old turns, keep the recent ones verbatim",
            ),
            ("full", "summarize the whole conversation into a brief"),
            ("elide", "drop old tool output, no model call"),
        ],
    },
    CommandSpec {
        name: "retry",
        args: "",
        help: "re-run your last message",
        arg_values: &[],
    },
    CommandSpec {
        name: "edit",
        args: "",
        help: "load your last message into the input line to edit and resend",
        arg_values: &[],
    },
    CommandSpec {
        name: "undo",
        args: "",
        help: "revert the file changes from the last turn",
        arg_values: &[],
    },
    CommandSpec {
        name: "commit",
        args: "",
        help: "stage all changes and commit them (git add -A && git commit)",
        arg_values: &[],
    },
    CommandSpec {
        name: "version",
        args: "",
        help: "show version",
        arg_values: &[],
    },
    CommandSpec {
        name: "export",
        args: "[path]",
        help: "export the conversation to a file (default: transcript.md)",
        arg_values: &[],
    },
    CommandSpec {
        name: "mcp",
        args: "",
        help: "inspect the MCP endpoint (server, tools, model count)",
        arg_values: &[],
    },
    CommandSpec {
        name: "hf",
        args: "<search|menu|files|download> ...",
        help: "discover and download Hugging Face Hub model files",
        arg_values: &[
            ("search", "search Hub models"),
            ("menu", "list downloadable repos for a Hub author"),
            ("author", "alias for menu"),
            ("user", "alias for menu"),
            ("files", "list repo files; accepts a menu number"),
            (
                "download",
                "download one repo file; accepts menu/file numbers",
            ),
            (
                "--",
                "with download, fetch every file in a repo or active author menu and delete after each",
            ),
            (
                "--keep",
                "with download, keep every file in a repo or active author menu",
            ),
        ],
    },
    CommandSpec {
        name: "lsp",
        args: "[on|off|status]",
        help: "toggle the LSP subsystem, or show server status",
        arg_values: &[
            ("on", "enable LSP"),
            ("off", "disable LSP"),
            ("status", "show per-language server state"),
        ],
    },
    CommandSpec {
        name: "delegate",
        args: "[on|off|status]",
        help: "toggle the write-capable delegate subagent (off by default)",
        arg_values: &[
            ("on", "let the model delegate implementation subtasks"),
            ("off", "disable delegate"),
            ("status", "show whether delegate is on"),
        ],
    },
    CommandSpec {
        name: "dashboard",
        args: "[status|resume <id>]",
        help: "control a fleet: dispatch, monitor, and steer multiple agents (TUI)",
        arg_values: &[
            ("status", "list this project's resumable fleet sessions"),
            (
                "resume",
                "re-adopt a fleet session as a live row (most recent if no id)",
            ),
        ],
    },
    CommandSpec {
        name: "loop",
        args: "[<interval> <prompt>|list|cancel <id>]",
        help: "recurring agent turns on a cadence (60s..7d; auto-expire after 7 days)",
        arg_values: &[
            ("list", "show active loops"),
            ("cancel", "stop a loop by id: /loop cancel <id>"),
        ],
    },
    CommandSpec {
        name: "watch",
        args: "",
        help: "full-screen live dashboard of all active loops",
        arg_values: &[],
    },
    CommandSpec {
        name: "digest",
        args: "",
        help: "what your loops noticed, grouped — with what's new since you last looked",
        arg_values: &[],
    },
    CommandSpec {
        name: "status",
        args: "[topic]",
        help: "show runtime status, or discuss codebase status with a topic",
        arg_values: &[],
    },
    CommandSpec {
        name: "log",
        args: "",
        help: "write a local debug log for this session",
        arg_values: &[],
    },
    CommandSpec {
        name: "clear",
        args: "",
        help: "start a fresh conversation",
        arg_values: &[],
    },
    CommandSpec {
        name: "exit",
        args: "",
        help: "quit",
        arg_values: &[],
    },
];

/// The message `/init` runs as a turn: explore the project and write a concise
/// `HI.md` guide that future sessions load as context.
pub const INIT_PROMPT: &str = "Explore this project (use the list and read tools) and write a \
file named HI.md at the repository root — a concise guide for a coding agent working here. \
Cover: what the project is and does; the main directories and files and their roles; the exact \
build, test, lint, and run commands; and any conventions or constraints worth knowing. Be \
factual and tight — this file is loaded as context for future sessions. Create HI.md with the \
write tool, then end with a one-line summary of what you captured.";

/// Commands whose canonical name starts with `prefix` (case-insensitive), in
/// display order — drives the `/`-completion menu. An empty prefix lists all.
pub fn matching(prefix: &str) -> Vec<&'static CommandSpec> {
    let needle = prefix.to_lowercase();
    COMMANDS
        .iter()
        .filter(|c| c.name.starts_with(&needle))
        .collect()
}

/// Enumerable argument values (value, hint) for command `name` whose value
/// starts with `prefix` (case-insensitive) — drives argument completion in the
/// `/`-menu (e.g. `/compact ` → hybrid/full/elide). Empty when the command is
/// unknown, takes a freeform argument, or nothing matches.
pub fn arg_matching(name: &str, prefix: &str) -> Vec<(&'static str, &'static str)> {
    let needle = prefix.to_lowercase();
    COMMANDS
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(name))
        .map(|c| {
            c.arg_values
                .iter()
                .filter(|(v, _)| v.starts_with(&needle))
                .copied()
                .collect()
        })
        .unwrap_or_default()
}

/// Help text, generated from [`COMMANDS`] so it always lists exactly what
/// exists. Includes a keybindings section so Ctrl- shortcuts aren't secret.
pub fn help_text() -> String {
    let mut out = String::from("commands:\n");
    for c in COMMANDS {
        let left = if c.args.is_empty() {
            format!("/{}", c.name)
        } else {
            format!("/{} {}", c.name, c.args)
        };
        out.push_str(&format!("  {left:<18} {}\n", c.help));
    }
    out.push_str("aliases: /m /st /cp /redo /revert /new /changes /usage /debug /h /?");
    out.push_str("\n\nkeybindings (TUI):\n");
    out.push_str("  Ctrl-T             toggle reasoning (thinking) collapse\n");
    out.push_str("  Ctrl-D             toggle the working-tree diff panel\n");
    out.push_str("  Ctrl-?             toggle the agent observability panel\n");
    out.push_str("  Ctrl-C             interrupt the running turn; double-press idle to quit\n");
    out.push_str("  Ctrl-R             fuzzy-search input history\n");
    out.push_str("  Ctrl-A / Ctrl-E    move cursor to start / end of the line\n");
    out.push_str("  Ctrl-U             clear the input line\n");
    out.push_str("  Alt-Enter          insert a newline (multi-line prompt)\n");
    out.push_str("  PageUp / PageDown  scroll the transcript\n");
    out.push_str("  Esc                clear input or dismiss panels\n");
    out.push_str("  /quit              quit\n");
    out
}

#[cfg(test)]
mod tests {
    use super::{
        COMMANDS, Command, GoalLimitArg, expand_prompt_macro, goal_arg_is_objective, help_text,
        matching, parse, parse_goal_limit,
    };

    #[test]
    fn every_listed_command_parses_to_a_real_command() {
        // Guards against the menu/help listing a command no frontend can run.
        for spec in COMMANDS {
            let line = format!("/{}", spec.name);
            match parse(&line) {
                Some(Command::Unknown(_)) | None => {
                    panic!("listed command {line} does not parse")
                }
                Some(_) => {}
            }
        }
    }

    #[test]
    fn help_text_matches_non_quitting_idle_keybindings() {
        let help = help_text();
        assert!(
            help.contains("Ctrl-D             toggle the working-tree diff panel"),
            "Ctrl-D should be documented as diff toggle:\n{help}"
        );
        assert!(
            help.contains("double-press idle to quit"),
            "Ctrl-C should document idle quit behavior:\n{help}"
        );
        assert!(
            help.contains("Esc                clear input or dismiss panels"),
            "Esc should not be documented as idle quit:\n{help}"
        );
        assert!(
            !help.contains("Ctrl-D (idle)") && !help.contains("quit when the line is empty"),
            "stale quit bindings should not be advertised:\n{help}"
        );
    }

    #[test]
    fn matching_filters_by_prefix() {
        // Empty prefix → everything; a prefix narrows; no match → empty.
        assert_eq!(matching("").len(), COMMANDS.len());
        let m = matching("co");
        assert!(m.iter().any(|c| c.name == "compact"));
        assert!(m.iter().any(|c| c.name == "copy"));
        assert!(m.iter().all(|c| c.name.starts_with("co")));
        assert!(matching("zzz").is_empty());
    }

    #[test]
    fn arg_matching_filters_enumerable_values() {
        use super::arg_matching;
        fn names(v: Vec<(&'static str, &'static str)>) -> Vec<&'static str> {
            v.into_iter().map(|(n, _)| n).collect()
        }
        // Empty prefix → all of the command's values, in order.
        assert_eq!(
            names(arg_matching("compact", "")),
            ["hybrid", "full", "elide"]
        );
        // A prefix narrows; case-insensitive.
        assert_eq!(names(arg_matching("compact", "h")), ["hybrid"]);
        assert_eq!(names(arg_matching("compact", "E")), ["elide"]);
        // No match, freeform-arg command, and unknown command all → empty.
        assert!(arg_matching("compact", "z").is_empty());
        assert!(arg_matching("model", "").is_empty());
        assert!(arg_matching("nope", "").is_empty());
    }

    #[test]
    fn every_compact_kind_value_parses() {
        // The menu's compact values must stay in lockstep with the parser, or the
        // menu would offer a kind /compact can't actually run.
        let compact = COMMANDS.iter().find(|c| c.name == "compact").unwrap();
        for (value, _) in compact.arg_values {
            assert!(
                crate::CompactionKind::from_arg(value).is_some(),
                "compact kind {value:?} listed in the menu must parse"
            );
        }
    }

    #[test]
    fn parses_commands_and_ignores_plain_input() {
        assert_eq!(parse("hello there"), None);
        assert_eq!(parse("/help"), Some(Command::Help));
        assert_eq!(parse("  /q "), Some(Command::Quit));
        assert_eq!(
            parse("/model gpt-4o"),
            Some(Command::Model("gpt-4o".into()))
        );
        assert_eq!(parse("/model"), Some(Command::Model(String::new())));
        assert_eq!(
            parse("/moa fix this"),
            Some(Command::Moa("fix this".into()))
        );
        assert_eq!(
            parse("/provider sonnet"),
            Some(Command::Provider("sonnet".into()))
        );
        assert_eq!(parse("/provider"), Some(Command::Provider(String::new())));
        assert_eq!(
            parse("/prov local"),
            Some(Command::Provider("local".into()))
        );
        assert_eq!(
            parse("/provider add"),
            Some(Command::Provider("add".into()))
        );
        assert_eq!(
            parse("/provider edit sonnet"),
            Some(Command::Provider("edit sonnet".into()))
        );
        assert_eq!(
            parse("/provider remove local"),
            Some(Command::Provider("remove local".into()))
        );
        assert_eq!(
            parse("/provider rm local"),
            Some(Command::Provider("rm local".into()))
        );
        assert_eq!(
            parse("/verify cargo test"),
            Some(Command::Verify("cargo test".into()))
        );
        assert_eq!(parse("/verify"), Some(Command::Verify(String::new())));
        assert_eq!(parse("/status"), Some(Command::Status));
        assert!(matches!(
            parse("/status codebase state"),
            Some(Command::Prompt(_))
        ));
        assert!(matches!(
            parse("/review scheduler"),
            Some(Command::Prompt(_))
        ));
        assert!(matches!(
            parse("/security unsafe unwraps"),
            Some(Command::Prompt(_))
        ));
        assert!(matches!(
            parse("/audit token leaks"),
            Some(Command::Prompt(_))
        ));
        assert!(matches!(
            parse("/roadmap next work"),
            Some(Command::Prompt(_))
        ));
        assert!(matches!(
            parse("/gaps missing pieces"),
            Some(Command::Prompt(_))
        ));
        assert_eq!(parse("/log"), Some(Command::Log));
        assert_eq!(parse("/diff"), Some(Command::Diff));
        assert_eq!(parse("/copy"), Some(Command::Copy(String::new())));
        assert_eq!(parse("/copy all"), Some(Command::Copy("all".into())));
        assert_eq!(parse("/goal"), Some(Command::Goal(String::new())));
        assert_eq!(
            parse("/goal ship it"),
            Some(Command::Goal("ship it".into()))
        );
        assert_eq!(parse("/init"), Some(Command::Init));
        assert_eq!(
            parse("/learn fix this"),
            Some(Command::Learn("fix this".into()))
        );
        assert_eq!(parse("/learn"), Some(Command::Learn(String::new())));
        assert_eq!(parse("/skills"), Some(Command::Skills));
        assert_eq!(
            parse("/skill release-flow"),
            Some(Command::Skill("release-flow".into()))
        );
        assert_eq!(parse("/compact"), Some(Command::Compact(String::new())));
        assert_eq!(
            parse("/compact hybrid"),
            Some(Command::Compact("hybrid".into()))
        );
        assert_eq!(parse("/redo"), Some(Command::Retry));
        assert_eq!(parse("/undo"), Some(Command::Undo));
        assert_eq!(parse("/bogus"), Some(Command::Unknown("bogus".into())));
        // `/lsp` parses with an optional arg.
        assert_eq!(parse("/lsp"), Some(Command::Lsp(String::new())));
        assert_eq!(parse("/lsp on"), Some(Command::Lsp("on".into())));
        assert_eq!(parse("/lsp off"), Some(Command::Lsp("off".into())));
        // `/delegate` toggles the write subagent.
        assert_eq!(parse("/dashboard"), Some(Command::Dashboard(String::new())));
        assert_eq!(parse("/fleet"), Some(Command::Dashboard(String::new())));
        assert_eq!(
            parse("/fleet status"),
            Some(Command::Dashboard("status".into()))
        );
        assert_eq!(parse("/delegate"), Some(Command::Delegate(String::new())));
        assert_eq!(parse("/delegate on"), Some(Command::Delegate("on".into())));
        assert_eq!(
            parse("/delegate off"),
            Some(Command::Delegate("off".into()))
        );
        assert_eq!(
            parse("/hf search llama"),
            Some(Command::Hf("search llama".into()))
        );
        assert_eq!(
            parse("/hd download --"),
            Some(Command::Hf("download --".into()))
        );
        // Removed `/tokens` aliases redirect to a hint instead of bare "unknown".
        assert!(matches!(
            parse("/usage"),
            Some(Command::Removed(m)) if m.contains("removed")
        ));
        assert!(matches!(
            parse("/cost"),
            Some(Command::Removed(m)) if m.contains("removed")
        ));
    }

    #[test]
    fn loop_interval_and_arg_parsing() {
        use super::{LoopArg, parse_loop_arg, parse_loop_interval};
        // Units + bounds (60s..7d), bare number = seconds.
        assert_eq!(parse_loop_interval("60s"), Some(60));
        assert_eq!(parse_loop_interval("90"), Some(90));
        assert_eq!(parse_loop_interval("30m"), Some(1800));
        assert_eq!(parse_loop_interval("2h"), Some(7200));
        assert_eq!(parse_loop_interval("1d"), Some(86_400));
        assert_eq!(parse_loop_interval("7d"), Some(7 * 86_400));
        assert_eq!(parse_loop_interval("59s"), None, "below the 60s floor");
        assert_eq!(parse_loop_interval("8d"), None, "above the 7d ceiling");
        assert_eq!(parse_loop_interval("2w"), None);
        assert_eq!(parse_loop_interval(""), None);
        // Arg splitting.
        assert_eq!(parse_loop_arg(""), LoopArg::List);
        assert_eq!(parse_loop_arg("list"), LoopArg::List);
        assert_eq!(parse_loop_arg("cancel 3"), LoopArg::Cancel(3));
        assert_eq!(parse_loop_arg("cancel #3"), LoopArg::Cancel(3));
        assert_eq!(
            parse_loop_arg("30m watch the CI logs"),
            LoopArg::Create {
                secs: 1800,
                prompt: "watch the CI logs".into()
            }
        );
        assert!(matches!(parse_loop_arg("30m"), LoopArg::Invalid(_)));
        assert!(matches!(
            parse_loop_arg("fast check ci"),
            LoopArg::Invalid(_)
        ));
        assert!(matches!(parse_loop_arg("cancel abc"), LoopArg::Invalid(_)));
        // Pause / resume / budget.
        assert_eq!(parse_loop_arg("pause 3"), LoopArg::Pause(3));
        assert_eq!(parse_loop_arg("resume #3"), LoopArg::Resume(3));
        assert_eq!(
            parse_loop_arg("budget 3 500k"),
            LoopArg::Budget {
                id: 3,
                tokens: Some(500_000)
            }
        );
        assert_eq!(
            parse_loop_arg("budget 3 off"),
            LoopArg::Budget {
                id: 3,
                tokens: None
            }
        );
        assert!(matches!(
            parse_loop_arg("budget 3 nope"),
            LoopArg::Invalid(_)
        ));
        assert!(matches!(parse_loop_arg("budget"), LoopArg::Invalid(_)));
        // Triggers (`on <id> <cmd|off>`).
        assert_eq!(
            parse_loop_arg("on 3 notify-send hi"),
            LoopArg::Trigger {
                id: 3,
                cmd: Some("notify-send hi".into())
            }
        );
        assert_eq!(
            parse_loop_arg("on 3 off"),
            LoopArg::Trigger { id: 3, cmd: None }
        );
        assert!(matches!(parse_loop_arg("on"), LoopArg::Invalid(_)));
        // `once …` must not be mistaken for an `on` trigger.
        assert!(!matches!(
            parse_loop_arg("once in a while check"),
            LoopArg::Trigger { .. }
        ));
        // Auto-fix toggle (`fix <id> on|pr|off`).
        assert_eq!(
            parse_loop_arg("fix 3 on"),
            LoopArg::Fix {
                id: 3,
                on: true,
                pr: false
            }
        );
        assert_eq!(
            parse_loop_arg("fix 3 pr"),
            LoopArg::Fix {
                id: 3,
                on: true,
                pr: true
            }
        );
        assert_eq!(
            parse_loop_arg("fix #3 off"),
            LoopArg::Fix {
                id: 3,
                on: false,
                pr: false
            }
        );
        assert!(matches!(parse_loop_arg("fix 3 maybe"), LoopArg::Invalid(_)));
        assert!(matches!(parse_loop_arg("fix"), LoopArg::Invalid(_)));
        // Fire windows (`window <id> <H-H [weekdays]|off>`).
        assert_eq!(
            parse_loop_arg("window 3 9-17"),
            LoopArg::Window {
                id: 3,
                window: Some((9, 17, false))
            }
        );
        assert_eq!(
            parse_loop_arg("window 3 9-17 weekdays"),
            LoopArg::Window {
                id: 3,
                window: Some((9, 17, true))
            }
        );
        assert_eq!(
            parse_loop_arg("window 3 off"),
            LoopArg::Window {
                id: 3,
                window: None
            }
        );
        assert!(matches!(
            parse_loop_arg("window 3 25-30"),
            LoopArg::Invalid(_)
        ));
        assert!(matches!(
            parse_loop_arg("window 3 nope"),
            LoopArg::Invalid(_)
        ));
        assert_eq!(parse_loop_arg("cost"), LoopArg::Cost);
        // Window parse edge cases.
        assert_eq!(super::parse_loop_window("0-24"), Some((0, 24, false)));
        assert_eq!(
            super::parse_loop_window("22-6"),
            Some((22, 6, false)),
            "wrap ok"
        );
        assert_eq!(super::parse_loop_window("9-9"), None, "empty window");
        // Token-count parsing.
        assert_eq!(super::parse_token_count("500k"), Some(500_000));
        assert_eq!(super::parse_token_count("1.5m"), Some(1_500_000));
        assert_eq!(super::parse_token_count("250000"), Some(250_000));
        assert_eq!(super::parse_token_count("nope"), None);
        // Command parse.
        assert_eq!(parse("/loop"), Some(Command::Loop(String::new())));
        assert_eq!(parse("/watch"), Some(Command::Watch));
        assert_eq!(parse("/digest"), Some(Command::Digest));
        assert_eq!(parse("/activity"), Some(Command::Digest));
        assert_eq!(
            parse("/loop 30m check ci"),
            Some(Command::Loop("30m check ci".into()))
        );
    }

    #[test]
    fn goal_arg_routing_and_limit_parsing() {
        // Objectives go to the planner; control subcommands do not.
        assert!(goal_arg_is_objective("port this service to Rust"));
        assert!(goal_arg_is_objective("limitless refactor")); // not a `limit` subcommand
        for control in [
            "", "  ", "clear", "off", "none", "pause", "resume", "limit", "limit 20",
        ] {
            assert!(
                !goal_arg_is_objective(control),
                "control arg routed as objective: {control:?}"
            );
        }
        // Limit parsing.
        assert_eq!(parse_goal_limit("limit 20"), Some(GoalLimitArg::Set(20)));
        assert_eq!(parse_goal_limit("limit"), Some(GoalLimitArg::Show));
        assert_eq!(parse_goal_limit("limit off"), Some(GoalLimitArg::Unlimited));
        assert_eq!(
            parse_goal_limit("limit none"),
            Some(GoalLimitArg::Unlimited)
        );
        assert_eq!(parse_goal_limit("limit 0"), Some(GoalLimitArg::Unlimited));
        assert_eq!(
            parse_goal_limit("limit huge"),
            Some(GoalLimitArg::Invalid("huge".into()))
        );
        // Not a limit subcommand → None (handled elsewhere).
        assert_eq!(parse_goal_limit("port to Rust"), None);
        assert_eq!(parse_goal_limit("limitless"), None);
    }

    #[test]
    fn prompt_macros_expand_to_read_only_inspection_prompts() {
        let review = expand_prompt_macro("/review parser").unwrap();
        assert!(review.contains("Read-only review request"));
        assert!(review.contains("parser"));
        assert!(review.contains("Do not write"));
        assert!(review.contains("Use read-only inspection"));

        let security = expand_prompt_macro("/security unsafe unwraps").unwrap();
        assert!(security.contains("unsafe unwraps"));
        assert!(security.contains("unsafe"));
        assert!(security.contains("unwrap"));
        assert!(security.contains("secret/token/auth"));

        let audit = expand_prompt_macro("/audit token leaks").unwrap();
        assert!(audit.contains("Read-only security request"));
        assert!(audit.contains("token leaks"));
        assert!(audit.contains("secret/token/auth"));

        let status = expand_prompt_macro("/status codebase state").unwrap();
        assert!(status.contains("codebase state"));
        assert!(status.contains("git status/diff"));

        let build = expand_prompt_macro("/build gpu training calculator").unwrap();
        assert!(build.contains("Build gpu training calculator."));
        assert!(build.contains("Inspect the workspace"));
        assert!(build.contains("Edit or create"));
        assert!(build.contains("validation command"));
        assert!(build.contains("changed files and validation commands"));

        assert!(expand_prompt_macro("/status").is_none());
    }
}
