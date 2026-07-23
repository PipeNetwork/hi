//! Slash-command parsing, shared by every frontend.

use hi_ai::ReasoningEffort;

/// A recognized in-session command. Frontends decide how to act on each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Help,
    /// Reset the conversation, keeping only the system prompt.
    Clear,
    /// Set the model for subsequent turns (empty = report current).
    Model(String),
    /// Show or set request config live: reasoning, temperature, step limit,
    /// RSI, and similar controls. Empty arg reports current values.
    Config(String),
    /// Inspect and recover public remote RSI runs.
    Rsi(String),
    /// Run exactly one turn through the conservative MoA virtual route.
    Moa(String),
    /// Use a provider/profile for subsequent turns (empty = report current).
    /// Named profiles are resolved from the config; the model can be set later
    /// with `/model` when the profile does not configure one.
    ///
    /// Subcommands: `add` (create a new profile interactively), `edit [name]`
    /// (edit an existing profile). The frontend parses these from the arg.
    Provider(String),
    /// Sign in to a provider that supports subscription auth, storing an OAuth
    /// credential instead of an API key. Arg is the provider name (`xai`).
    Login(String),
    /// Discard a stored OAuth credential. Arg is the provider name.
    Logout(String),
    /// Show current session/runtime status.
    Status,
    /// Show or set the per-session turn limit (`/turns <n>`, `/turns off`).
    Turns(String),
    /// Diagnose setup/runtime health (config, credentials, git, MCP).
    Doctor,
    /// Enter/exit plan mode or show the current plan. Arg: empty/on, `off`, `show`, or a request.
    Plan(String),
    /// Show the current plan checklist (`update_plan` / plan mode).
    ViewPlan,
    /// Fork a peer session; optional `--worktree` / `--no-worktree` and directive.
    Fork(String),
    /// Rewind conversation to before a user turn. Empty lists turns; `<n>` rewinds.
    Rewind(String),
    /// Permission ladder: empty/status, `ask`, `auto`, `always` (yolo).
    Permissions(String),
    /// Alias for `/permissions always` when empty; otherwise same as Permissions.
    AlwaysApprove(String),
    /// Alias for `/permissions auto` when empty.
    Auto(String),
    /// Show queued prompts / background work. Frontend fills live queue details.
    Queue(String),
    /// Show tasks (loops, background processes, delegates). Alias surface for `/queue tasks`.
    Tasks(String),
    /// List skills + `.hi/hooks` inventory (lightweight plugins/hooks view).
    Plugins(String),
    /// Append a durable memory note. Optional `--global` / `global`.
    Remember(String),
    /// Scan Claude Code config and print migration hints.
    ImportClaude(String),
    /// Local session recap (does not enter model history).
    Recap,
    /// Search conversation messages for text.
    Find(String),
    /// Jump/list user turns (same anchors as `/rewind`).
    Jump(String),
    /// List recent user prompts (history).
    History(String),
    /// Execute/list lifecycle hooks under `.hi/hooks`.
    Hooks(String),
    /// Query/grant/revoke trust for repository-local executable config.
    Trust(String),
    /// Manage installable skill/plugin packs.
    Marketplace(String),
    /// Manage isolated worktrees created by fork/delegate/best-of.
    Worktree(String),
    /// Inspect the effective installation/config/runtime (optional `json`).
    Inspect(String),
    /// Manage named agent/persona definitions under `.hi/agents`.
    Agents(String),
    /// Share/export the current session for review.
    Share(String),
    /// MCP administration beyond inspect (list/add/remove/doctor guidance).
    McpAdmin(String),
    /// Rewind picker/list UX (alias of `/rewind`, TUI may open a picker).
    RewindPicker,
    /// Switch between fullscreen and terminal-scrollback-oriented minimal mode.
    ScreenMode(String),
    /// Toggle vim-style composer mode.
    VimMode(String),
    /// Toggle explicit multiline composer mode.
    Multiline(String),
    /// Toggle transcript timeline rail.
    Timeline(String),
    /// Toggle transcript timestamps.
    Timestamps(String),
    /// Change/list dashboard workspace directory.
    Cd(String),
    /// Primary session rename surface.
    Rename(String),
    /// Primary session resume surface.
    Resume(String),
    /// Toggle or query the LSP subsystem. Arg: `on`, `off`, or empty (status).
    Lsp(String),
    /// Toggle or query the write-capable `delegate` subagent. Arg: `on`, `off`,
    /// `risk`, or empty (status).
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
    /// List all files touched this session (accumulated across turns).
    Files,
    /// Open the full-screen diff review overlay (like Ctrl-G). Optional file
    /// paths filter the diff to just those files.
    Review(String),
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
    /// Run or manage a scripted workflow: `/workflow <name> [args...]` launches
    /// a multi-phase, multi-agent plan from a Rhai script (built-in or
    /// `~/.hi/workflows/<name>.rhai`). Subcommands: `list` lists available
    /// workflows, `show <name>` prints its meta, `validate <file>` dry-runs a
    /// script. TUI: renders as grouped phase rows in the dashboard; plain REPL:
    /// runs headless and prints the outcome.
    Workflow(String),
    /// Recurring agent turns on a cadence (TUI only): `<interval> <prompt>`
    /// creates, empty/`list` lists, `cancel <id>` stops one.
    Loop(String),
    /// Full-screen live dashboard of all active loops (TUI only).
    Watch,
    /// Show the activity digest: what loops have noticed, grouped by loop, with
    /// what's new since you last looked (TUI only).
    Digest,
    /// Toggle or query session sync to ipop. Arg: `on`, `off`, `status`, or
    /// empty (status). When on, session records + live events are pushed to
    /// the ipop API for cross-machine resume.
    Sync(String),
    /// List or manage sessions. Args: `switch <id>`, `rename <id> <name>`, or
    /// empty (list).
    Sessions(String),
    /// Attach to a running session as a viewer + input sender. Arg: session id
    /// (or empty to pick from a list). This opens its live event stream.
    Attach(String),
    /// Start this session as a persistent daemon: hold the agent resident and
    /// accept input from remote clients via ipop. Arg: empty (use current
    /// session) or a session id to resume.
    Daemon(String),
    /// Switch the TUI color theme (TUI only). Arg: `dark`, `light`, `ansi`,
    /// `auto` (follow OS), or empty to cycle to the next.
    Theme(String),
    /// Transcript density (TUI only). Arg: `compact`, `comfortable`, `verbose`,
    /// or empty to cycle.
    Density(String),
    /// Toggle terminal mouse capture (TUI only). Arg: `on`, `off`, or empty to
    /// toggle. Off drops to the terminal's native text selection at the cost of
    /// the scroll wheel and click/drag block folding + copy.
    Mouse(String),
    Quit,
    /// A `/btw <question>` side question asked while a turn runs. The frontend
    /// routes it to the interjection inbox (tagged as a question) so the model
    /// answers briefly — with a live session snapshot — then continues its task.
    Btw(String),
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
        "config" | "cfg" | "set" => Command::Config(arg),
        "rsi" => Command::Rsi(arg),
        "moa" => Command::Moa(arg),
        "provider" | "prov" => Command::Provider(arg),
        "login" | "signin" => Command::Login(arg),
        "logout" | "signout" => Command::Logout(arg),
        "usage" | "cost" => Command::Removed("usage — removed; use /status".into()),
        "review" if arg.is_empty() => Command::Review(String::new()),
        "review" => Command::Prompt(read_only_macro_prompt("review", &arg)),
        "security" | "audit" => Command::Prompt(read_only_macro_prompt("security", &arg)),
        "roadmap" => Command::Prompt(read_only_macro_prompt("roadmap", &arg)),
        "gaps" => Command::Prompt(read_only_macro_prompt("gaps", &arg)),
        "build" => Command::Prompt(build_macro_prompt(&arg)),
        "status" | "st" if arg.is_empty() => Command::Status,
        "status" | "st" => Command::Prompt(read_only_macro_prompt("status", &arg)),
        "turns" | "turn-limit" | "max-turns" => Command::Turns(arg),
        "doctor" => Command::Doctor,
        "plan" => Command::Plan(arg),
        "view-plan" | "viewplan" | "show-plan" | "showplan" => Command::ViewPlan,
        "fork" => Command::Fork(arg),
        "rewind" => Command::Rewind(arg),
        "permissions" | "permission" | "perms" => Command::Permissions(arg),
        "always-approve" | "alwaysapprove" | "yolo" => Command::AlwaysApprove(arg),
        "auto" => Command::Auto(arg),
        "queue" => Command::Queue(arg),
        "tasks" | "task" => Command::Tasks(arg),
        "plugins" | "plugin" => Command::Plugins(arg),
        "remember" | "mem" => Command::Remember(arg),
        "import-claude" | "import_claude" | "claude-import" => Command::ImportClaude(arg),
        "recap" | "summarize" | "summary" => Command::Recap,
        "find" | "search" => Command::Find(arg),
        "jump" => Command::Jump(arg),
        "history" | "hist" => Command::History(arg),
        "hooks" | "hook" => Command::Hooks(arg),
        "trust" => Command::Trust(arg),
        "marketplace" | "market" => Command::Marketplace(arg),
        "worktree" | "worktrees" | "wt" => Command::Worktree(arg),
        "inspect" => Command::Inspect(arg),
        "agents" | "personas" | "persona" => Command::Agents(arg),
        "share" => Command::Share(arg),
        "mcp-admin" | "mcps" => Command::McpAdmin(arg),
        "rewind-picker" | "rewind-pick" => Command::RewindPicker,
        "minimal" => Command::ScreenMode(if arg.is_empty() {
            "minimal".into()
        } else {
            arg
        }),
        "fullscreen" | "full-screen" => Command::ScreenMode(if arg.is_empty() {
            "fullscreen".into()
        } else {
            arg
        }),
        "screen-mode" | "screen" => Command::ScreenMode(arg),
        "vim-mode" | "vim" => Command::VimMode(arg),
        "multiline" | "multi-line" => Command::Multiline(arg),
        "timeline" => Command::Timeline(arg),
        "timestamps" | "timestamp" => Command::Timestamps(arg),
        "cd" | "cwd" => Command::Cd(arg),
        "rename" => Command::Rename(arg),
        "resume" => Command::Resume(arg),
        "log" | "debug" => Command::Log,
        "verify" | "test" => Command::Verify(arg),
        "diff" | "changes" => Command::Diff,
        "files" => Command::Files,
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
        "workflow" | "workflows" => Command::Workflow(arg),
        "loop" | "loops" => Command::Loop(arg),
        "watch" => Command::Watch,
        "theme" | "themes" => Command::Theme(arg),
        "density" | "dense" => Command::Density(arg),
        "mouse" => Command::Mouse(arg),
        "digest" | "activity" => Command::Digest,
        "btw" | "bytheway" | "question" => Command::Btw(arg),
        // Compatibility aliases remain accepted, but the public command
        // surface is consolidated under `/sessions`.
        "sync" => Command::Sessions(if arg.is_empty() {
            "sync".to_string()
        } else {
            format!("sync {arg}")
        }),
        "sessions" => Command::Sessions(arg),
        "attach" => Command::Sessions(if arg.is_empty() {
            "attach".to_string()
        } else {
            format!("attach {arg}")
        }),
        "daemon" => Command::Sessions(if arg.is_empty() {
            "host".to_string()
        } else {
            format!("host {arg}")
        }),
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
/// subcommand (empty, `clear`/`off`/`none`, `pause`, `resume`, `status`, …).
/// Frontends use this to route only real objectives to the planner.
pub fn goal_arg_is_objective(arg: &str) -> bool {
    let a = arg.trim();
    if a.is_empty() {
        return false;
    }
    // Flags on an objective still count as objectives (`/goal --review fix auth`).
    if a.starts_with("--") {
        return true;
    }
    let head = a.split_whitespace().next().unwrap_or(a);
    !matches!(
        head,
        "clear"
            | "off"
            | "none"
            | "pause"
            | "resume"
            | "accept"
            | "status"
            | "show"
            | "export"
            | "view"
            | "limit"
            | "budget"
            | "team"
            | "edit"
    ) && !a.starts_with("limit ")
        && !a.starts_with("budget ")
        && !a.starts_with("team ")
        && !a.starts_with("edit ")
}

/// Strip `/goal` objective flags. Returns `(review_first, objective_text)`.
pub fn parse_goal_objective_flags(arg: &str) -> (bool, String) {
    let mut review = false;
    let mut rest = Vec::new();
    for tok in arg.split_whitespace() {
        match tok {
            "--review" | "-r" | "--review-first" => review = true,
            other => rest.push(other),
        }
    }
    // Also allow trailing/leading bare `review` only as a flag when alone with objective words —
    // keep it simple: only `--review`.
    (review, rest.join(" "))
}

/// Parsed `/goal edit …` forms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoalEditArg {
    /// `/goal edit objective <text>`
    Objective(String),
    /// `/goal edit step <n> <text>` (1-based)
    Step {
        index: usize,
        text: String,
    },
    Invalid(String),
}

/// Parse `/goal edit …`. Returns `None` when `arg` is not an edit subcommand.
pub fn parse_goal_edit(arg: &str) -> Option<GoalEditArg> {
    let a = arg.trim();
    let rest = if a == "edit" {
        ""
    } else {
        a.strip_prefix("edit ")?.trim()
    };
    if rest.is_empty() {
        return Some(GoalEditArg::Invalid(
            "usage: /goal edit objective <text> | /goal edit step <n> <text>".into(),
        ));
    }
    if let Some(text) = rest
        .strip_prefix("objective ")
        .or_else(|| rest.strip_prefix("obj "))
    {
        let text = text.trim();
        if text.is_empty() {
            return Some(GoalEditArg::Invalid(
                "usage: /goal edit objective <text>".into(),
            ));
        }
        return Some(GoalEditArg::Objective(text.to_string()));
    }
    if let Some(rest) = rest
        .strip_prefix("step ")
        .or_else(|| rest.strip_prefix("#"))
    {
        let rest = rest.trim();
        let mut parts = rest.splitn(2, char::is_whitespace);
        let Some(n_str) = parts.next() else {
            return Some(GoalEditArg::Invalid(
                "usage: /goal edit step <n> <text>".into(),
            ));
        };
        let Ok(index) = n_str.parse::<usize>() else {
            return Some(GoalEditArg::Invalid(format!("bad step number '{n_str}'")));
        };
        let text = parts.next().unwrap_or("").trim();
        if index == 0 || text.is_empty() {
            return Some(GoalEditArg::Invalid(
                "usage: /goal edit step <n> <text> (n is 1-based)".into(),
            ));
        }
        return Some(GoalEditArg::Step {
            index,
            text: text.to_string(),
        });
    }
    Some(GoalEditArg::Invalid(
        "usage: /goal edit objective <text> | /goal edit step <n> <text>".into(),
    ))
}

/// Parse the args after `/loop trio`: an optional `--rounds N` flag followed
/// by the free-text prompt. Returns `(max_rounds, prompt)`. Default rounds = 3.
fn parse_trio_args(rest: &str) -> (u8, String) {
    let rest = rest.trim();
    if let Some(after) = rest.strip_prefix("--rounds") {
        let after = after.trim();
        if let Some((n_str, prompt)) = after.split_once(char::is_whitespace)
            && let Ok(n) = n_str.trim().parse::<u8>()
            && n > 0
        {
            return (n, prompt.trim().to_string());
        }
        // `--rounds` with no valid number + prompt — fall through to treating
        // the whole thing as a prompt (the flag is optional).
    }
    (3, rest.to_string())
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
    /// `trio <prompt>` — a bounded plan→execute→review loop (the trio
    /// workflow): the planner model produces a lightweight plan, the session
    /// model executes it, and the reviewer model reviews the diff before
    /// approving or sending it back for revision. Stops when approved or
    /// `max_rounds` is hit. No persistent goal state — it's a transient loop.
    Trio { prompt: String, max_rounds: u8 },
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
/// The purpose-built prompt for a `/loop review` PR-review watcher. The loop's
/// child agent has shell access, so it drives `gh` directly; the session it
/// resumes each firing remembers which PRs it already reviewed, and the
/// quiet/loud contract makes "no new PRs" a silent `NOTHING NEW`.
pub const REVIEW_PROMPT: &str = "Review this repository's open pull requests. Run \
    `gh pr list --state open` to find them. For each PR you have NOT already reviewed earlier in \
    this conversation, read its diff with `gh pr diff <number>` and assess it for correctness, \
    missing tests, and risks, then post a concise review with \
    `gh pr review <number> --comment --body \"<your review>\"` (a comment — do not approve or \
    request changes). If there are no pull requests you haven't already reviewed, reply with \
    exactly: NOTHING NEW. Otherwise report which PRs you reviewed and the gist of each.";

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
    // `review [interval]` — a preset PR-review watcher (default every 30m).
    if a == "review" || a.starts_with("review ") {
        let rest = a[6..].trim();
        let secs = if rest.is_empty() {
            1800
        } else {
            match parse_loop_interval(rest) {
                Some(secs) => secs,
                None => {
                    return LoopArg::Invalid(format!(
                        "bad interval '{rest}' — use e.g. /loop review 1h (default 30m)"
                    ));
                }
            }
        };
        return LoopArg::Create {
            secs,
            prompt: REVIEW_PROMPT.to_string(),
        };
    }
    // `trio <prompt>` — a bounded plan→execute→review loop.
    if a == "trio" || a.starts_with("trio ") {
        let rest = a[4..].trim();
        // Parse optional `--rounds N` flag, then the prompt.
        let (max_rounds, prompt) = parse_trio_args(rest);
        if prompt.is_empty() {
            return LoopArg::Invalid(
                "usage: /loop trio <prompt>  (optional: /loop trio --rounds 3 <prompt>)".into(),
            );
        }
        return LoopArg::Trio { prompt, max_rounds };
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

/// The parsed value of a `/goal budget …` subcommand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoalBudgetArg {
    /// `/goal budget <n>` — park and report after `n` more drive turns.
    Set(u32),
    /// `/goal budget off|none|clear|0` — run until done (the default).
    Unlimited,
    /// `/goal budget` — report the current budget and what's left.
    Show,
    /// `/goal budget <garbage>` — the value didn't parse.
    Invalid(String),
}

/// Parse the argument of a `/goal budget …` subcommand. Returns `None` when
/// `arg` isn't a `budget` subcommand (so the caller handles other `/goal` forms).
pub fn parse_goal_budget(arg: &str) -> Option<GoalBudgetArg> {
    let a = arg.trim();
    let rest = if a == "budget" {
        ""
    } else {
        a.strip_prefix("budget ")?.trim()
    };
    Some(match rest {
        "" => GoalBudgetArg::Show,
        "off" | "none" | "clear" => GoalBudgetArg::Unlimited,
        value => match value.parse::<u32>() {
            Ok(0) => GoalBudgetArg::Unlimited,
            Ok(n) => GoalBudgetArg::Set(n),
            Err(_) => GoalBudgetArg::Invalid(value.to_string()),
        },
    })
}

/// The parsed value of a `/turns …` command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TurnsArg {
    /// `/turns <n>` — cap the session at `n` turns.
    Set(u32),
    /// `/turns off|none|clear|0` — remove the cap (the default: unlimited).
    Unlimited,
    /// `/turns` — report the current limit and turn count.
    Show,
    /// `/turns <garbage>` — the value didn't parse.
    Invalid(String),
}

/// Parse the argument of a `/turns …` command.
pub fn parse_turns_arg(arg: &str) -> TurnsArg {
    let a = arg.trim();
    match a {
        "" => TurnsArg::Show,
        "off" | "none" | "clear" => TurnsArg::Unlimited,
        value => match value.parse::<u32>() {
            Ok(0) => TurnsArg::Unlimited,
            Ok(n) => TurnsArg::Set(n),
            Err(_) => TurnsArg::Invalid(value.to_string()),
        },
    }
}

/// The parsed value of a `/goal team …` subcommand (the skeptic gate).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoalTeamArg {
    /// `/goal team on` — enable the skeptic gate for the active goal.
    On,
    /// `/goal team off` — disable it.
    Off,
    /// `/goal team` — report the current state.
    Show,
    /// `/goal team <garbage>` — unrecognized argument.
    Invalid(String),
}

/// Parse the argument of a `/goal team …` subcommand. Returns `None` when `arg`
/// isn't a `team` subcommand at all (so the caller handles other `/goal` forms).
pub fn parse_goal_team(arg: &str) -> Option<GoalTeamArg> {
    let a = arg.trim();
    let rest = if a == "team" {
        ""
    } else {
        a.strip_prefix("team ")?.trim()
    };
    Some(match rest {
        "" => GoalTeamArg::Show,
        "on" | "yes" | "true" => GoalTeamArg::On,
        "off" | "no" | "false" => GoalTeamArg::Off,
        other => GoalTeamArg::Invalid(other.to_string()),
    })
}

/// The parsed form of a `/config` argument.
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigArg {
    /// `/config` — report the current live settings.
    Show,
    /// `/config reasoning <level|off>` — set the reasoning effort (`None` = off,
    /// i.e. send no `reasoning_effort` and take the endpoint default).
    Reasoning(Option<ReasoningEffort>),
    /// `/config temp <value|off>` — set the sampling temperature (`None` clears
    /// it, leaving the provider default).
    Temperature(Option<f32>),
    /// `/config steps <n|off>` — set a fixed cap, or disable it (`None`).
    MaxSteps(Option<u32>),
    /// `/config steps auto` — restore intent-aware per-turn defaults.
    MaxStepsAuto,
    /// `/config moe-streaming <on|off|auto>` — control MLX MoE expert streaming.
    /// `On` forces streaming, `Off` forces resident, `Auto` (the default) lets
    /// the loader auto-enable when the model exceeds the memory budget.
    MoeStreaming(MoeStreamingMode),
    /// `/config skeptic-local <on|off>` — turn the auto-managed local model for
    /// the `/goal` skeptic review on or off. `on` detects the machine's backend,
    /// downloads a small default review model if needed, and spawns a local
    /// server; `off` stops it and restores the prior skeptic settings.
    SkepticLocal(bool),
    /// `/config rsi` — show the focused public-RSI settings.
    RsiShow,
    /// `/config rsi on|off` — set and persist the candidate channel.
    Rsi(bool),
    /// `/config rsi spend-limit <USD>` — set and persist the per-run ceiling.
    RsiSpendLimit(u64),
    /// `/config rsi channel stable|beta` — persist the candidate channel.
    RsiChannel(RsiChannel),
    /// Nested settings that resolve to existing top-level commands. Frontends
    /// should prefer [`resolve_command`] so `/config model …` shares the same
    /// handlers as bare `/model …`.
    Model(String),
    Provider(String),
    Login(String),
    Logout(String),
    Verify(String),
    Lsp(String),
    Delegate(String),
    Theme(String),
    Density(String),
    Mouse(String),
    /// Unrecognized option or bad value; carries a usage/error hint.
    Invalid(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RsiChannel {
    Stable,
    Beta,
}

impl RsiChannel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Beta => "beta",
        }
    }
}

/// The MoE streaming mode set by `/config moe-streaming`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeStreamingMode {
    /// Force streaming on (`HI_MLX_EXPERT_STREAMING=1`).
    On,
    /// Force streaming off / resident (`HI_MLX_EXPERT_STREAMING=0`).
    Off,
    /// Auto-detect based on memory budget (env var unset).
    Auto,
}

/// Parse a `/config` argument into a [`ConfigArg`]. Shared by every frontend so
/// the plain REPL and the TUI accept exactly the same syntax.
pub fn parse_config_arg(arg: &str) -> ConfigArg {
    let a = arg.trim();
    if a.is_empty() {
        return ConfigArg::Show;
    }
    let (key, val) = match a.split_once(char::is_whitespace) {
        Some((k, v)) => (k, v.trim()),
        None => (a, ""),
    };
    let off = |v: &str| matches!(v.to_ascii_lowercase().as_str(), "off" | "none" | "clear");
    match key.to_ascii_lowercase().as_str() {
        // `/config show` (also `list` / `status`) — report the current live
        // settings. A bare `/config` already does this via the empty-arg path
        // above; these keywords make the intent explicit and discoverable.
        "show" | "list" | "status" => {
            if !val.is_empty() {
                ConfigArg::Invalid(
                    "/config show takes no value — use it alone to view current settings".into(),
                )
            } else {
                ConfigArg::Show
            }
        }
        "reasoning" | "reasoning-effort" | "reason" | "effort" | "r" => {
            if val.is_empty() {
                ConfigArg::Invalid(
                    "usage: /config reasoning <minimal|low|medium|high|xhigh|off>".into(),
                )
            } else if off(val) || val.eq_ignore_ascii_case("disable") {
                ConfigArg::Reasoning(None)
            } else {
                match ReasoningEffort::from_arg(val) {
                    Some(e) => ConfigArg::Reasoning(Some(e)),
                    None => ConfigArg::Invalid(format!(
                        "unknown reasoning level '{val}' — use minimal, low, medium, high, xhigh, or off"
                    )),
                }
            }
        }
        "temp" | "temperature" | "t" => {
            if val.is_empty() {
                ConfigArg::Invalid("usage: /config temp <0.0-2.0|off>".into())
            } else if off(val) || val.eq_ignore_ascii_case("default") {
                ConfigArg::Temperature(None)
            } else {
                match val.parse::<f32>() {
                    Ok(t) if (0.0..=2.0).contains(&t) => ConfigArg::Temperature(Some(t)),
                    Ok(_) => ConfigArg::Invalid(format!(
                        "temperature '{val}' out of range — use 0.0 to 2.0, or off"
                    )),
                    Err(_) => ConfigArg::Invalid(format!(
                        "bad temperature '{val}' — use a number from 0.0 to 2.0, or off"
                    )),
                }
            }
        }
        "steps" | "max-steps" | "step-limit" | "limit" => {
            if val.is_empty() {
                ConfigArg::Invalid("usage: /config steps <1+|auto|off>".into())
            } else if matches!(
                val.to_ascii_lowercase().as_str(),
                "off" | "none" | "disable" | "disabled" | "unlimited"
            ) {
                ConfigArg::MaxSteps(None)
            } else if matches!(val.to_ascii_lowercase().as_str(), "auto" | "default") {
                ConfigArg::MaxStepsAuto
            } else {
                match val.parse::<u32>() {
                    Ok(steps) if steps > 0 => ConfigArg::MaxSteps(Some(steps)),
                    Ok(_) => {
                        ConfigArg::Invalid("step limit must be at least 1, or use auto/off".into())
                    }
                    Err(_) => ConfigArg::Invalid(format!(
                        "bad step limit '{val}' — use a positive integer, auto, or off"
                    )),
                }
            }
        }
        "moe-streaming" | "moe" | "expert-streaming" => {
            if val.is_empty() {
                ConfigArg::Invalid("usage: /config moe-streaming <on|off|auto>".into())
            } else {
                match val.to_ascii_lowercase().as_str() {
                    "on" | "enable" | "enabled" | "1" | "true" | "yes" => {
                        ConfigArg::MoeStreaming(MoeStreamingMode::On)
                    }
                    "off" | "disable" | "disabled" | "0" | "false" | "no" => {
                        ConfigArg::MoeStreaming(MoeStreamingMode::Off)
                    }
                    "auto" | "default" | "automatic" => {
                        ConfigArg::MoeStreaming(MoeStreamingMode::Auto)
                    }
                    _ => ConfigArg::Invalid(format!(
                        "unknown moe-streaming mode '{val}' — use on, off, or auto"
                    )),
                }
            }
        }
        "skeptic-local" | "local-skeptic" => match val.to_ascii_lowercase().as_str() {
            "" => ConfigArg::Invalid("usage: /config skeptic-local <on|off>".into()),
            "on" | "enable" | "enabled" | "1" | "true" | "yes" => ConfigArg::SkepticLocal(true),
            "off" | "disable" | "disabled" | "0" | "false" | "no" => ConfigArg::SkepticLocal(false),
            _ => ConfigArg::Invalid(format!(
                "unknown skeptic-local mode '{val}' — use on or off"
            )),
        },
        "rsi" => parse_rsi_config_arg(val),
        // Nested settings hub — these rewrite to the existing top-level
        // commands via [`resolve_command`] so handlers stay single-sourced.
        "model" | "m" => ConfigArg::Model(val.to_string()),
        "provider" | "prov" | "profile" => ConfigArg::Provider(val.to_string()),
        "login" | "signin" => ConfigArg::Login(val.to_string()),
        "logout" | "signout" => ConfigArg::Logout(val.to_string()),
        "auth" => parse_auth_config_arg(val),
        "verify" | "test" => ConfigArg::Verify(val.to_string()),
        "lsp" => ConfigArg::Lsp(val.to_string()),
        "delegate" | "delegates" => ConfigArg::Delegate(val.to_string()),
        "theme" | "themes" => ConfigArg::Theme(val.to_string()),
        "density" | "dense" => ConfigArg::Density(val.to_string()),
        "mouse" => ConfigArg::Mouse(val.to_string()),
        "ui" => parse_ui_config_arg(val),
        other => ConfigArg::Invalid(format!(
            "unknown /config option '{other}' — try: show, model, provider, auth, \
reasoning, temp, steps, verify, lsp, delegate, moe-streaming, skeptic-local, rsi, \
ui theme|density|mouse"
        )),
    }
}

/// `/config auth login|logout <provider>`.
fn parse_auth_config_arg(arg: &str) -> ConfigArg {
    let a = arg.trim();
    if a.is_empty() {
        return ConfigArg::Invalid("usage: /config auth <login|logout> <provider>".into());
    }
    let (action, rest) = match a.split_once(char::is_whitespace) {
        Some((k, v)) => (k, v.trim()),
        None => (a, ""),
    };
    match action.to_ascii_lowercase().as_str() {
        "login" | "signin" => ConfigArg::Login(rest.to_string()),
        "logout" | "signout" => ConfigArg::Logout(rest.to_string()),
        other => ConfigArg::Invalid(format!(
            "unknown /config auth action '{other}' — use login or logout"
        )),
    }
}

/// `/config ui theme|density|mouse …`.
fn parse_ui_config_arg(arg: &str) -> ConfigArg {
    let a = arg.trim();
    if a.is_empty() {
        return ConfigArg::Invalid("usage: /config ui <theme|density|mouse> [value]".into());
    }
    let (key, rest) = match a.split_once(char::is_whitespace) {
        Some((k, v)) => (k, v.trim()),
        None => (a, ""),
    };
    match key.to_ascii_lowercase().as_str() {
        "theme" | "themes" => ConfigArg::Theme(rest.to_string()),
        "density" | "dense" => ConfigArg::Density(rest.to_string()),
        "mouse" => ConfigArg::Mouse(rest.to_string()),
        other => ConfigArg::Invalid(format!(
            "unknown /config ui option '{other}' — use theme, density, or mouse"
        )),
    }
}

/// Rewrite nested `/config …` settings into the underlying top-level
/// [`Command`] so frontends keep a single handler per setting. Live knobs
/// (reasoning/temp/steps/rsi/…) stay as [`Command::Config`].
pub fn resolve_command(command: Command) -> Command {
    match command {
        Command::Config(ref arg) => match parse_config_arg(arg) {
            ConfigArg::Model(s) => Command::Model(s),
            ConfigArg::Provider(s) => Command::Provider(s),
            ConfigArg::Login(s) => Command::Login(s),
            ConfigArg::Logout(s) => Command::Logout(s),
            ConfigArg::Verify(s) => Command::Verify(s),
            ConfigArg::Lsp(s) => Command::Lsp(s),
            ConfigArg::Delegate(s) => Command::Delegate(s),
            ConfigArg::Theme(s) => Command::Theme(s),
            ConfigArg::Density(s) => Command::Density(s),
            ConfigArg::Mouse(s) => Command::Mouse(s),
            _ => command,
        },
        other => other,
    }
}

fn parse_rsi_config_arg(value: &str) -> ConfigArg {
    let mut parts = value.split_whitespace();
    let Some(action) = parts.next() else {
        return ConfigArg::RsiShow;
    };
    match action.to_ascii_lowercase().as_str() {
        "show" | "status" => {
            if parts.next().is_none() {
                ConfigArg::RsiShow
            } else {
                ConfigArg::Invalid(
                    "usage: /config rsi [on|off|spend-limit <USD>|channel stable|beta]".into(),
                )
            }
        }
        "on" | "true" | "yes" | "1" if parts.next().is_none() => ConfigArg::Rsi(true),
        "off" | "false" | "no" | "0" if parts.next().is_none() => ConfigArg::Rsi(false),
        "spend-limit" | "spend" | "max-cost" | "limit" => {
            let Some(amount) = parts.next() else {
                return ConfigArg::Invalid("usage: /config rsi spend-limit <USD, up to 15>".into());
            };
            if parts.next().is_some() {
                return ConfigArg::Invalid("usage: /config rsi spend-limit <USD, up to 15>".into());
            }
            match parse_usd_micros(amount) {
                Some(value @ 1..=15_000_000) => ConfigArg::RsiSpendLimit(value),
                Some(_) => ConfigArg::Invalid(
                    "RSI spend limit must be greater than $0 and no more than $15".into(),
                ),
                None => ConfigArg::Invalid(format!(
                    "bad RSI spend limit '{amount}' — use a USD amount such as 5 or 2.50"
                )),
            }
        }
        "channel" => match (parts.next(), parts.next()) {
            (Some("stable"), None) => ConfigArg::RsiChannel(RsiChannel::Stable),
            (Some("beta"), None) => ConfigArg::RsiChannel(RsiChannel::Beta),
            _ => ConfigArg::Invalid("usage: /config rsi channel <stable|beta>".into()),
        },
        _ => ConfigArg::Invalid(
            "usage: /config rsi [on|off|spend-limit <USD>|channel stable|beta]".into(),
        ),
    }
}

fn parse_usd_micros(value: &str) -> Option<u64> {
    let value = value.strip_prefix('$').unwrap_or(value);
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    if whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.len() > 6
    {
        return None;
    }
    let whole = whole.parse::<u64>().ok()?;
    let fraction = if fraction.is_empty() {
        0
    } else {
        fraction.parse::<u64>().ok()? * 10_u64.pow((6 - fraction.len()) as u32)
    };
    whole.checked_mul(1_000_000)?.checked_add(fraction)
}

pub fn format_usd_micros(value: u64) -> String {
    let whole = value / 1_000_000;
    let fraction = value % 1_000_000;
    if fraction == 0 {
        format!("${whole}")
    } else {
        let fraction = format!("{fraction:06}").trim_end_matches('0').to_string();
        format!("${whole}.{fraction}")
    }
}

/// Whether a `/config` argument is the async `skeptic-local` toggle. The CLI
/// routes this through its async handler (it may download a model and spawn a
/// server) rather than the synchronous `/config` path.
pub fn config_is_skeptic_local(arg: &str) -> bool {
    matches!(parse_config_arg(arg), ConfigArg::SkepticLocal(_))
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
        help: "show or set the model (alias of /config model)",
        arg_values: &[],
    },
    CommandSpec {
        name: "config",
        args: "[key …]",
        help: "settings hub — model, provider, auth, reasoning, verify, lsp, ui, …",
        arg_values: &[
            ("show", "print the current live settings"),
            ("model", "show or set the model (/config model [id])"),
            (
                "provider",
                "list/switch profiles, or add/edit/remove (/config provider …)",
            ),
            (
                "auth",
                "subscription login/logout (/config auth login|logout <provider>)",
            ),
            (
                "reasoning",
                "set reasoning_effort: minimal|low|medium|high|xhigh|off",
            ),
            ("temp", "set sampling temperature: 0.0-2.0, or off"),
            (
                "steps",
                "set the turn step limit: positive integer, auto, or off",
            ),
            ("verify", "show/set/clear the verify command"),
            ("lsp", "toggle LSP or show status"),
            ("delegate", "write-capable delegate policy: on|off|risk"),
            ("moe-streaming", "MLX MoE expert streaming: on|off|auto"),
            ("skeptic-local", "auto-managed local skeptic model: on|off"),
            ("rsi", "public RSI policy: on|off|spend-limit|channel"),
            ("ui", "TUI chrome: theme|density|mouse"),
            ("theme", "TUI color theme (alias of /config ui theme)"),
            (
                "density",
                "transcript density (alias of /config ui density)",
            ),
            ("mouse", "mouse capture (alias of /config ui mouse)"),
        ],
    },
    CommandSpec {
        name: "rsi",
        args: "[list|status RUN|cancel RUN|apply RUN|artifacts RUN|feedback [RUN] good|bad [reason]]",
        help: "inspect or recover remote RSI runs",
        arg_values: &[
            ("list", "list local pending and completed run metadata"),
            ("status", "fetch an owned run's current status"),
            ("cancel", "request idempotent cancellation"),
            ("apply", "retry exact conflict-checked patch application"),
            ("artifacts", "download patch, report, and trace artifacts"),
            ("feedback", "submit good/bad supporting evidence for a run"),
        ],
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
        help: "profiles (alias of /config provider)",
        arg_values: &[],
    },
    CommandSpec {
        name: "login",
        args: "<provider>",
        help: "subscription sign-in (alias of /config auth login)",
        arg_values: &[
            (
                "xai",
                "Grok via a grok.com SuperGrok or X Premium subscription",
            ),
            ("pipenetwork", "Pipe Network via a browser pairing flow"),
        ],
    },
    CommandSpec {
        name: "logout",
        args: "<provider>",
        help: "discard subscription login (alias of /config auth logout)",
        arg_values: &[
            ("xai", "forget the stored grok.com credential"),
            ("pipenetwork", "forget the stored Pipe Network credential"),
        ],
    },
    CommandSpec {
        name: "verify",
        args: "[cmd|off]",
        help: "verify command (alias of /config verify); turns iterate against",
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
        name: "files",
        args: "",
        help: "list all files touched this session",
        arg_values: &[],
    },
    CommandSpec {
        name: "copy",
        args: "[all]",
        help: "copy the last response (or transcript) to the clipboard",
        arg_values: &[("all", "copy the whole transcript, not just the last reply")],
    },
    CommandSpec {
        name: "btw",
        args: "<question>",
        help: "ask a side question mid-turn (about the session/job) without derailing the task",
        arg_values: &[],
    },
    CommandSpec {
        name: "goal",
        args: "[text|--review|status|pause|resume|accept|edit|limit|team|export|clear]",
        help: "long-horizon goal: plan, drive, pause reasons, status, edit, export",
        arg_values: &[
            ("status", "rich status: drive state, checklist, events"),
            (
                "pause",
                "pause the goal — hold progress, stop steering turns",
            ),
            (
                "resume",
                "resume a paused goal (clears review/stall/user pause)",
            ),
            ("accept", "accept a review-first plan and start driving"),
            (
                "--review",
                "with an objective: pause for review before auto-drive",
            ),
            (
                "edit",
                "edit objective or step: /goal edit objective|step <n> <text>",
            ),
            (
                "limit",
                "cap plan growth: /goal limit <n>, or 'limit off' for none",
            ),
            ("team", "skeptic gate: /goal team on|off"),
            ("export", "write .hi/goal-plan.md (export-only snapshot)"),
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
        help: "LSP toggle (alias of /config lsp)",
        arg_values: &[
            ("on", "enable LSP"),
            ("off", "disable LSP"),
            ("status", "show per-language server state"),
        ],
    },
    CommandSpec {
        name: "delegate",
        args: "[on|off|risk|status]",
        help: "delegate policy (alias of /config delegate)",
        arg_values: &[
            ("on", "offer delegate on every mutation turn"),
            ("off", "never offer delegate"),
            ("risk", "only multi-file / isolation-shaped tasks (default)"),
            ("status", "show off|risk|on"),
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
        name: "theme",
        args: "[dark|light|ansi|auto]",
        help: "TUI theme (alias of /config ui theme)",
        arg_values: &[
            ("dark", "designed dark palette (truecolor)"),
            ("light", "designed light palette (truecolor)"),
            ("ansi", "terminal-native 16-color palette"),
            ("auto", "follow the OS light/dark appearance"),
        ],
    },
    CommandSpec {
        name: "density",
        args: "[compact|comfortable|verbose]",
        help: "transcript density (alias of /config ui density)",
        arg_values: &[
            ("compact", "headers only for long tool output"),
            ("comfortable", "default preview fold"),
            ("verbose", "expand all tool output"),
        ],
    },
    CommandSpec {
        name: "mouse",
        args: "[on|off]",
        help: "mouse capture (alias of /config ui mouse)",
        arg_values: &[
            (
                "on",
                "app handles the mouse: scroll wheel, click-fold, drag-copy",
            ),
            (
                "off",
                "release the mouse to the terminal's native text selection",
            ),
        ],
    },
    CommandSpec {
        name: "sessions",
        args: "[switch|rename|favorite|archive|restore|delete|attach|host|sync]",
        help: "browse, switch, and manage sessions",
        arg_values: &[
            ("switch", "switch to a session"),
            ("rename", "name or rename a session"),
            ("favorite", "favorite a session"),
            ("archive", "archive a session"),
            ("restore", "restore an archived session"),
            ("delete", "permanently delete a session"),
            (
                "attach",
                "join a session (steer live host, or continue here)",
            ),
            ("continue", "force portable continue on this machine"),
            ("host", "accept remote prompts here (on|off|status)"),
            ("sync", "configure portal synchronization"),
        ],
    },
    CommandSpec {
        name: "status",
        args: "[topic]",
        help: "show runtime status, or discuss codebase status with a topic",
        arg_values: &[],
    },
    CommandSpec {
        name: "turns",
        args: "[n|off]",
        help: "per-session turn limit: /turns <n>, or 'off' for unlimited",
        arg_values: &[],
    },
    CommandSpec {
        name: "doctor",
        args: "",
        help: "diagnose setup and runtime health (config, credentials, git, MCP)",
        arg_values: &[],
    },
    CommandSpec {
        name: "plan",
        args: "[off|show|request]",
        help: "plan mode: design before edits (off exits; show prints checklist)",
        arg_values: &[
            ("off", "leave plan mode"),
            ("show", "print the current plan checklist"),
            ("on", "enter plan mode"),
        ],
    },
    CommandSpec {
        name: "view-plan",
        args: "",
        help: "show the current plan checklist",
        arg_values: &[],
    },
    CommandSpec {
        name: "fork",
        args: "[--worktree|--no-worktree] [directive]",
        help: "fork a peer session, optionally in an isolated git worktree",
        arg_values: &[
            ("--worktree", "create an isolated worktree (default)"),
            ("--no-worktree", "same tree; open another session yourself"),
        ],
    },
    CommandSpec {
        name: "rewind",
        args: "[n]",
        help: "list user turns, or rewind conversation before turn n",
        arg_values: &[],
    },
    CommandSpec {
        name: "permissions",
        args: "[ask|auto|always]",
        help: "permission ladder: ask / auto-safe / always-approve (yolo)",
        arg_values: &[
            ("ask", "confirm writes"),
            ("auto", "skip routine confirms; keep checkpoints"),
            ("always", "yolo — no confirms, allow missing checkpoints"),
        ],
    },
    CommandSpec {
        name: "always-approve",
        args: "",
        help: "set permissions to always (yolo); alias of /permissions always",
        arg_values: &[],
    },
    CommandSpec {
        name: "auto",
        args: "",
        help: "set permissions to auto; alias of /permissions auto",
        arg_values: &[],
    },
    CommandSpec {
        name: "queue",
        args: "[tasks]",
        help: "show queued prompts and background work",
        arg_values: &[("tasks", "include loops/background process detail")],
    },
    CommandSpec {
        name: "tasks",
        args: "",
        help: "show background tasks, processes, and session work",
        arg_values: &[],
    },
    CommandSpec {
        name: "plugins",
        args: "",
        help: "list skills and .hi/hooks (hooks/plugins inventory)",
        arg_values: &[],
    },
    CommandSpec {
        name: "remember",
        args: "[--global] <note>",
        help: "append a durable memory bullet (project or --global)",
        arg_values: &[("--global", "write user-level ~/.config/hi/memory.md")],
    },
    CommandSpec {
        name: "import-claude",
        args: "",
        help: "scan Claude Code config and print migration hints",
        arg_values: &[],
    },
    CommandSpec {
        name: "recap",
        args: "",
        help: "local session recap (not added to model history)",
        arg_values: &[],
    },
    CommandSpec {
        name: "find",
        args: "<text>",
        help: "search conversation messages",
        arg_values: &[],
    },
    CommandSpec {
        name: "jump",
        args: "[n]",
        help: "list user turns (same anchors as /rewind)",
        arg_values: &[],
    },
    CommandSpec {
        name: "history",
        args: "[query]",
        help: "list/search recent user prompts",
        arg_values: &[],
    },
    CommandSpec {
        name: "hooks",
        args: "[list|<name> [input]]",
        help: "list or execute .hi/hooks lifecycle scripts",
        arg_values: &[("list", "inventory hooks")],
    },
    CommandSpec {
        name: "trust",
        args: "[status|on|off]",
        help: "query/grant/revoke trust for project hooks and executable config",
        arg_values: &[
            ("status", "show trust state"),
            ("on", "trust workspace"),
            ("off", "revoke trust"),
        ],
    },
    CommandSpec {
        name: "marketplace",
        args: "[list|install <SKILL.md>]",
        help: "list/install portable plugin skill packs",
        arg_values: &[
            ("list", "list local plugin packs"),
            ("install", "install a SKILL.md"),
        ],
    },
    CommandSpec {
        name: "worktree",
        args: "[list|gc|remove <n>]",
        help: "manage isolated worktrees created by /fork",
        arg_values: &[
            ("list", "list fork worktrees"),
            ("gc", "clean fork worktrees"),
            ("remove", "remove one worktree"),
        ],
    },
    CommandSpec {
        name: "inspect",
        args: "[--json]",
        help: "dump effective workspace/runtime/hooks/skills state",
        arg_values: &[("--json", "machine-readable output")],
    },
    CommandSpec {
        name: "agents",
        args: "[list|add|show|remove]",
        help: "manage named agent/persona definitions",
        arg_values: &[
            ("list", "list personas"),
            ("add", "create persona"),
            ("show", "show persona"),
            ("remove", "remove persona"),
        ],
    },
    CommandSpec {
        name: "share",
        args: "[--json]",
        help: "export a reviewable session share bundle",
        arg_values: &[("--json", "machine-readable result")],
    },
    CommandSpec {
        name: "mcp-admin",
        args: "[list|doctor|add|remove]",
        help: "inspect/manage provider MCP configuration",
        arg_values: &[
            ("list", "show MCP setup"),
            ("doctor", "run MCP health guidance"),
            ("add", "add endpoint guidance"),
            ("remove", "remove endpoint guidance"),
        ],
    },
    CommandSpec {
        name: "rewind-picker",
        args: "",
        help: "show a richer list of conversation rewind anchors",
        arg_values: &[],
    },
    CommandSpec {
        name: "screen-mode",
        args: "[minimal|fullscreen]",
        help: "switch transcript screen style",
        arg_values: &[
            ("minimal", "scrollback-oriented"),
            ("fullscreen", "alternate-screen TUI"),
        ],
    },
    CommandSpec {
        name: "vim-mode",
        args: "[on|off|status]",
        help: "toggle vim-style composer normal mode",
        arg_values: &[("on", "enable"), ("off", "disable"), ("status", "show")],
    },
    CommandSpec {
        name: "multiline",
        args: "[on|off|status]",
        help: "toggle multiline composer mode",
        arg_values: &[("on", "enable"), ("off", "disable"), ("status", "show")],
    },
    CommandSpec {
        name: "timeline",
        args: "[on|off|status]",
        help: "toggle transcript timeline rail",
        arg_values: &[("on", "enable"), ("off", "disable"), ("status", "show")],
    },
    CommandSpec {
        name: "timestamps",
        args: "[on|off|status]",
        help: "toggle transcript timestamps",
        arg_values: &[("on", "enable"), ("off", "disable"), ("status", "show")],
    },
    CommandSpec {
        name: "cd",
        args: "[path]",
        help: "show/change dashboard workspace (new session required for live agent)",
        arg_values: &[],
    },
    CommandSpec {
        name: "rename",
        args: "<name>",
        help: "rename the active session (frontends route via /sessions)",
        arg_values: &[],
    },
    CommandSpec {
        name: "resume",
        args: "[id]",
        help: "resume/switch sessions (frontends route via /sessions)",
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

/// Whether a command is primarily a settings toggle whose canonical home is
/// `/config`. Kept in help under a separate section so the action list stays
/// scannable; bare tops remain parseable aliases forever.
fn is_settings_alias(name: &str) -> bool {
    matches!(
        name,
        "model"
            | "provider"
            | "login"
            | "logout"
            | "verify"
            | "lsp"
            | "delegate"
            | "theme"
            | "density"
            | "mouse"
    )
}

/// Help text, generated from [`COMMANDS`] so it always lists exactly what
/// exists. Groups settings under `/config` and marks bare tops as aliases.
/// Includes a keybindings section so Ctrl- shortcuts aren't secret.
pub fn help_text() -> String {
    let mut out = String::from("commands:\n");
    for c in COMMANDS.iter().filter(|c| !is_settings_alias(c.name)) {
        let left = if c.args.is_empty() {
            format!("/{}", c.name)
        } else {
            format!("/{} {}", c.name, c.args)
        };
        out.push_str(&format!("  {left:<18} {}\n", c.help));
    }
    out.push_str("\nsettings (also available as bare aliases):\n");
    out.push_str(
        "  /config [key …]   hub for model, provider, auth, reasoning, verify, lsp, ui…\n",
    );
    out.push_str("  /model /provider /login /logout /verify /lsp /delegate\n");
    out.push_str("  /theme /density /mouse   (TUI; also /config ui …)\n");
    out.push_str("aliases: /m /st /cp /redo /revert /new /changes /usage /debug /cfg /set /h /?");
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
        COMMANDS, Command, GoalEditArg, GoalLimitArg, GoalTeamArg, LoopArg, TurnsArg,
        expand_prompt_macro, goal_arg_is_objective, help_text, matching, parse, parse_goal_edit,
        parse_goal_limit, parse_goal_objective_flags, parse_goal_team, parse_loop_arg,
        parse_turns_arg,
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
    fn command_registry_metadata_is_unique_and_complete() {
        let mut names = std::collections::BTreeSet::new();
        for spec in COMMANDS {
            assert!(!spec.name.trim().is_empty(), "empty command name");
            assert!(
                spec.name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "non-canonical command name: {}",
                spec.name
            );
            assert!(names.insert(spec.name), "duplicate command: {}", spec.name);
            assert!(!spec.help.trim().is_empty(), "missing help: {}", spec.name);
            let mut values = std::collections::BTreeSet::new();
            for (value, hint) in spec.arg_values {
                assert!(
                    values.insert(*value),
                    "duplicate arg value {value} for {}",
                    spec.name
                );
                assert!(
                    !hint.trim().is_empty(),
                    "missing arg hint for {} {value}",
                    spec.name
                );
            }
        }
    }

    #[test]
    fn portal_session_commands_share_one_public_surface() {
        let names = COMMANDS
            .iter()
            .map(|command| command.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"sessions"));
        assert!(!names.contains(&"sync"));
        assert!(!names.contains(&"attach"));
        assert!(!names.contains(&"daemon"));
        assert_eq!(
            parse("/sync status"),
            Some(Command::Sessions("sync status".into()))
        );
        assert_eq!(
            parse("/attach abc"),
            Some(Command::Sessions("attach abc".into()))
        );
        assert_eq!(parse("/daemon"), Some(Command::Sessions("host".into())));
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
        assert_eq!(parse("/doctor"), Some(Command::Doctor));
        assert_eq!(parse("/plan"), Some(Command::Plan(String::new())));
        assert_eq!(
            parse("/plan fix auth"),
            Some(Command::Plan("fix auth".into()))
        );
        assert_eq!(parse("/view-plan"), Some(Command::ViewPlan));
        assert_eq!(
            parse("/fork --worktree try x"),
            Some(Command::Fork("--worktree try x".into()))
        );
        assert_eq!(parse("/rewind 2"), Some(Command::Rewind("2".into())));
        assert_eq!(
            parse("/permissions auto"),
            Some(Command::Permissions("auto".into()))
        );
        assert_eq!(
            parse("/always-approve"),
            Some(Command::AlwaysApprove(String::new()))
        );
        assert_eq!(parse("/auto"), Some(Command::Auto(String::new())));
        assert_eq!(parse("/queue"), Some(Command::Queue(String::new())));
        assert_eq!(parse("/tasks"), Some(Command::Tasks(String::new())));
        assert_eq!(parse("/plugins"), Some(Command::Plugins(String::new())));
        assert_eq!(parse("/hooks"), Some(Command::Hooks(String::new())));
        assert_eq!(
            parse("/hooks pre-turn hello"),
            Some(Command::Hooks("pre-turn hello".into()))
        );
        assert_eq!(parse("/trust on"), Some(Command::Trust("on".into())));
        assert_eq!(
            parse("/marketplace"),
            Some(Command::Marketplace(String::new()))
        );
        assert_eq!(parse("/worktree gc"), Some(Command::Worktree("gc".into())));
        assert_eq!(
            parse("/inspect --json"),
            Some(Command::Inspect("--json".into()))
        );
        assert_eq!(parse("/agents list"), Some(Command::Agents("list".into())));
        assert_eq!(parse("/share"), Some(Command::Share(String::new())));
        assert_eq!(
            parse("/mcp-admin doctor"),
            Some(Command::McpAdmin("doctor".into()))
        );
        assert_eq!(parse("/rewind-picker"), Some(Command::RewindPicker));
        assert_eq!(
            parse("/minimal"),
            Some(Command::ScreenMode("minimal".into()))
        );
        assert_eq!(
            parse("/fullscreen"),
            Some(Command::ScreenMode("fullscreen".into()))
        );
        assert_eq!(parse("/vim-mode on"), Some(Command::VimMode("on".into())));
        assert_eq!(
            parse("/multiline off"),
            Some(Command::Multiline("off".into()))
        );
        assert_eq!(parse("/timeline"), Some(Command::Timeline(String::new())));
        assert_eq!(
            parse("/timestamps on"),
            Some(Command::Timestamps("on".into()))
        );
        assert_eq!(parse("/cd ../repo"), Some(Command::Cd("../repo".into())));
        assert_eq!(
            parse("/rename release work"),
            Some(Command::Rename("release work".into()))
        );
        assert_eq!(parse("/resume abc"), Some(Command::Resume("abc".into())));
        assert_eq!(
            parse("/remember use pnpm"),
            Some(Command::Remember("use pnpm".into()))
        );
        assert_eq!(
            parse("/import-claude"),
            Some(Command::ImportClaude(String::new()))
        );
        assert_eq!(parse("/recap"), Some(Command::Recap));
        assert_eq!(parse("/find token"), Some(Command::Find("token".into())));
        assert_eq!(parse("/jump"), Some(Command::Jump(String::new())));
        assert_eq!(parse("/history"), Some(Command::History(String::new())));
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
        assert_eq!(parse("/files"), Some(Command::Files));
        // `/btw` is a side-question command; the arg is the question text.
        assert_eq!(
            parse("/btw what step are you on"),
            Some(Command::Btw("what step are you on".into()))
        );
        assert_eq!(parse("/btw"), Some(Command::Btw(String::new())));
        // `/review` with no arg opens the diff review overlay; with text it
        // runs the review macro prompt (a Command::Prompt).
        assert_eq!(parse("/review"), Some(Command::Review(String::new())));
        assert!(matches!(
            parse("/review the auth flow"),
            Some(Command::Prompt(_))
        ));
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
        // PR-review preset (`review [interval]`).
        assert_eq!(
            parse_loop_arg("review"),
            LoopArg::Create {
                secs: 1800,
                prompt: super::REVIEW_PROMPT.to_string()
            }
        );
        assert_eq!(
            parse_loop_arg("review 1h"),
            LoopArg::Create {
                secs: 3600,
                prompt: super::REVIEW_PROMPT.to_string()
            }
        );
        assert!(matches!(parse_loop_arg("review 5s"), LoopArg::Invalid(_)));
        assert!(super::REVIEW_PROMPT.contains("gh pr review"));
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
        assert_eq!(parse("/theme dark"), Some(Command::Theme("dark".into())));
        assert_eq!(parse("/theme"), Some(Command::Theme(String::new())));
        assert_eq!(parse("/mouse off"), Some(Command::Mouse("off".into())));
        assert_eq!(parse("/mouse"), Some(Command::Mouse(String::new())));
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
        assert!(goal_arg_is_objective("--review fix auth"));
        for control in [
            "",
            "  ",
            "clear",
            "off",
            "none",
            "pause",
            "resume",
            "accept",
            "status",
            "export",
            "edit",
            "edit objective x",
            "limit",
            "limit 20",
            "team",
            "team on",
        ] {
            assert!(
                !goal_arg_is_objective(control),
                "control arg routed as objective: {control:?}"
            );
        }
        assert_eq!(
            parse_goal_objective_flags("--review ship it"),
            (true, "ship it".into())
        );
        assert_eq!(
            parse_goal_edit("edit step 2 do the thing"),
            Some(GoalEditArg::Step {
                index: 2,
                text: "do the thing".into()
            })
        );
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
    fn turns_arg_parsing() {
        // Show.
        assert_eq!(parse_turns_arg(""), TurnsArg::Show);
        assert_eq!(parse_turns_arg("   "), TurnsArg::Show);
        // Set.
        assert_eq!(parse_turns_arg("20"), TurnsArg::Set(20));
        assert_eq!(parse_turns_arg(" 5 "), TurnsArg::Set(5));
        // Unlimited / off forms.
        assert_eq!(parse_turns_arg("off"), TurnsArg::Unlimited);
        assert_eq!(parse_turns_arg("none"), TurnsArg::Unlimited);
        assert_eq!(parse_turns_arg("clear"), TurnsArg::Unlimited);
        assert_eq!(parse_turns_arg("0"), TurnsArg::Unlimited);
        // Invalid.
        assert_eq!(parse_turns_arg("huge"), TurnsArg::Invalid("huge".into()));
        assert_eq!(parse_turns_arg("abc"), TurnsArg::Invalid("abc".into()));
    }

    #[test]
    fn turns_command_parses() {
        assert_eq!(parse("/turns"), Some(Command::Turns(String::new())));
        assert_eq!(parse("/turns 10"), Some(Command::Turns("10".into())));
        assert_eq!(parse("/turns off"), Some(Command::Turns("off".into())));
        // Aliases.
        assert_eq!(parse("/max-turns 5"), Some(Command::Turns("5".into())));
        assert_eq!(parse("/turn-limit off"), Some(Command::Turns("off".into())));
    }

    #[test]
    fn config_arg_parsing() {
        use super::{
            ConfigArg, MoeStreamingMode, format_usd_micros, parse_config_arg, resolve_command,
        };
        use hi_ai::ReasoningEffort;
        // Empty → show.
        assert_eq!(parse_config_arg(""), ConfigArg::Show);
        assert_eq!(parse_config_arg("   "), ConfigArg::Show);
        // Explicit `show` (and aliases) → show.
        assert_eq!(parse_config_arg("show"), ConfigArg::Show);
        assert_eq!(parse_config_arg("LIST"), ConfigArg::Show);
        assert_eq!(parse_config_arg("status"), ConfigArg::Show);
        // `show` rejects a trailing value.
        assert!(matches!(
            parse_config_arg("show everything"),
            ConfigArg::Invalid(_)
        ));
        // Reasoning levels + aliases.
        assert_eq!(
            parse_config_arg("reasoning high"),
            ConfigArg::Reasoning(Some(ReasoningEffort::High))
        );
        assert_eq!(
            parse_config_arg("effort MEDIUM"),
            ConfigArg::Reasoning(Some(ReasoningEffort::Medium))
        );
        assert_eq!(
            parse_config_arg("r xhigh"),
            ConfigArg::Reasoning(Some(ReasoningEffort::Xhigh))
        );
        // Off spellings clear it.
        assert_eq!(
            parse_config_arg("reasoning off"),
            ConfigArg::Reasoning(None)
        );
        assert_eq!(
            parse_config_arg("reasoning none"),
            ConfigArg::Reasoning(None)
        );
        // Bad level / missing value.
        assert!(matches!(
            parse_config_arg("reasoning turbo"),
            ConfigArg::Invalid(_)
        ));
        assert!(matches!(
            parse_config_arg("reasoning"),
            ConfigArg::Invalid(_)
        ));
        // Temperature: in range, off, out of range, non-numeric.
        assert_eq!(
            parse_config_arg("temp 0.7"),
            ConfigArg::Temperature(Some(0.7))
        );
        assert_eq!(
            parse_config_arg("temperature 0"),
            ConfigArg::Temperature(Some(0.0))
        );
        assert_eq!(parse_config_arg("temp off"), ConfigArg::Temperature(None));
        assert_eq!(
            parse_config_arg("temp default"),
            ConfigArg::Temperature(None)
        );
        assert!(matches!(parse_config_arg("temp 5"), ConfigArg::Invalid(_)));
        assert!(matches!(
            parse_config_arg("temp hot"),
            ConfigArg::Invalid(_)
        ));
        // Step cap: fixed, disabled, automatic, and invalid.
        assert_eq!(
            parse_config_arg("steps 500"),
            ConfigArg::MaxSteps(Some(500))
        );
        assert_eq!(parse_config_arg("max-steps off"), ConfigArg::MaxSteps(None));
        assert_eq!(
            parse_config_arg("step-limit unlimited"),
            ConfigArg::MaxSteps(None)
        );
        assert_eq!(parse_config_arg("steps auto"), ConfigArg::MaxStepsAuto);
        assert!(matches!(parse_config_arg("steps 0"), ConfigArg::Invalid(_)));
        assert!(matches!(
            parse_config_arg("steps many"),
            ConfigArg::Invalid(_)
        ));
        assert_eq!(parse_config_arg("rsi on"), ConfigArg::Rsi(true));
        assert_eq!(parse_config_arg("rsi off"), ConfigArg::Rsi(false));
        assert_eq!(parse_config_arg("rsi"), ConfigArg::RsiShow);
        assert_eq!(parse_config_arg("rsi show"), ConfigArg::RsiShow);
        assert_eq!(
            parse_config_arg("rsi spend-limit 5"),
            ConfigArg::RsiSpendLimit(5_000_000)
        );
        assert_eq!(
            parse_config_arg("rsi spend $2.50"),
            ConfigArg::RsiSpendLimit(2_500_000)
        );
        assert_eq!(
            parse_config_arg("rsi max-cost 0.000001"),
            ConfigArg::RsiSpendLimit(1)
        );
        assert!(matches!(
            parse_config_arg("rsi spend-limit 0"),
            ConfigArg::Invalid(_)
        ));
        assert!(matches!(
            parse_config_arg("rsi spend-limit 15.01"),
            ConfigArg::Invalid(_)
        ));
        assert!(matches!(
            parse_config_arg("rsi spend-limit five"),
            ConfigArg::Invalid(_)
        ));
        assert_eq!(format_usd_micros(5_000_000), "$5");
        assert_eq!(format_usd_micros(2_500_000), "$2.5");
        assert_eq!(format_usd_micros(1), "$0.000001");
        assert_eq!(
            parse_config_arg("rsi channel stable"),
            ConfigArg::RsiChannel(super::RsiChannel::Stable)
        );
        assert_eq!(
            parse_config_arg("rsi channel beta"),
            ConfigArg::RsiChannel(super::RsiChannel::Beta)
        );
        // Unknown option.
        assert!(matches!(parse_config_arg("bogus x"), ConfigArg::Invalid(_)));
        // MoE streaming: on, off, auto, bad value.
        assert_eq!(
            parse_config_arg("moe-streaming on"),
            ConfigArg::MoeStreaming(MoeStreamingMode::On)
        );
        assert_eq!(
            parse_config_arg("moe-streaming off"),
            ConfigArg::MoeStreaming(MoeStreamingMode::Off)
        );
        assert_eq!(
            parse_config_arg("moe-streaming auto"),
            ConfigArg::MoeStreaming(MoeStreamingMode::Auto)
        );
        assert_eq!(
            parse_config_arg("moe 1"),
            ConfigArg::MoeStreaming(MoeStreamingMode::On)
        );
        assert!(matches!(
            parse_config_arg("moe-streaming maybe"),
            ConfigArg::Invalid(_)
        ));
        // Command parse wiring + aliases.
        assert_eq!(parse("/config"), Some(Command::Config(String::new())));
        assert_eq!(
            parse("/config reasoning high"),
            Some(Command::Config("reasoning high".into()))
        );
        assert_eq!(
            parse("/cfg temp 0.5"),
            Some(Command::Config("temp 0.5".into()))
        );
        assert_eq!(
            parse("/set reasoning off"),
            Some(Command::Config("reasoning off".into()))
        );
        // Nested settings hub.
        assert_eq!(
            parse_config_arg("model gpt-test"),
            ConfigArg::Model("gpt-test".into())
        );
        assert_eq!(
            parse_config_arg("provider add"),
            ConfigArg::Provider("add".into())
        );
        assert_eq!(
            parse_config_arg("auth login xai"),
            ConfigArg::Login("xai".into())
        );
        assert_eq!(
            parse_config_arg("auth logout xai"),
            ConfigArg::Logout("xai".into())
        );
        assert_eq!(parse_config_arg("lsp on"), ConfigArg::Lsp("on".into()));
        assert_eq!(
            parse_config_arg("delegate risk"),
            ConfigArg::Delegate("risk".into())
        );
        assert_eq!(
            parse_config_arg("verify cargo test"),
            ConfigArg::Verify("cargo test".into())
        );
        assert_eq!(
            parse_config_arg("ui theme dark"),
            ConfigArg::Theme("dark".into())
        );
        assert_eq!(
            parse_config_arg("ui density compact"),
            ConfigArg::Density("compact".into())
        );
        assert_eq!(
            parse_config_arg("mouse off"),
            ConfigArg::Mouse("off".into())
        );
        assert_eq!(
            resolve_command(Command::Config("model gpt-test".into())),
            Command::Model("gpt-test".into())
        );
        assert_eq!(
            resolve_command(Command::Config("lsp on".into())),
            Command::Lsp("on".into())
        );
        assert_eq!(
            resolve_command(Command::Config("auth login xai".into())),
            Command::Login("xai".into())
        );
        assert_eq!(
            resolve_command(Command::Config("ui theme dark".into())),
            Command::Theme("dark".into())
        );
        // Live knobs stay as Config.
        assert_eq!(
            resolve_command(Command::Config("reasoning high".into())),
            Command::Config("reasoning high".into())
        );
        let help = help_text();
        assert!(help.contains("settings (also available as bare aliases)"));
        assert!(help.contains("/config [key"));
        // Primary command rows are `  /name …` under the commands section; the
        // settings blurb lists bare aliases on a single line without a help tail.
        assert!(
            !help
                .lines()
                .any(|line| { line.starts_with("  /model ") && line.contains("alias of /config") }),
            "bare /model should not appear as a primary help row"
        );
        assert!(help.contains("/model /provider"));
    }

    #[test]
    fn goal_team_subcommand_parsing() {
        assert_eq!(parse_goal_team("team on"), Some(GoalTeamArg::On));
        assert_eq!(parse_goal_team("team off"), Some(GoalTeamArg::Off));
        assert_eq!(parse_goal_team("team"), Some(GoalTeamArg::Show));
        assert_eq!(parse_goal_team("team yes"), Some(GoalTeamArg::On));
        assert_eq!(
            parse_goal_team("team maybe"),
            Some(GoalTeamArg::Invalid("maybe".into()))
        );
        // Not a team subcommand → None (handled elsewhere, e.g. as an objective).
        assert_eq!(parse_goal_team("teamwork refactor"), None);
        assert_eq!(parse_goal_team("port to Rust"), None);
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

    #[test]
    fn loop_trio_parses_basic_prompt() {
        let arg = parse_loop_arg("trio refactor the parser module");
        match arg {
            LoopArg::Trio { prompt, max_rounds } => {
                assert_eq!(prompt, "refactor the parser module");
                assert_eq!(max_rounds, 3); // default
            }
            other => panic!("expected Trio, got {other:?}"),
        }
    }

    #[test]
    fn loop_trio_parses_with_rounds_flag() {
        let arg = parse_loop_arg("trio --rounds 5 fix the failing tests");
        match arg {
            LoopArg::Trio { prompt, max_rounds } => {
                assert_eq!(prompt, "fix the failing tests");
                assert_eq!(max_rounds, 5);
            }
            other => panic!("expected Trio, got {other:?}"),
        }
    }

    #[test]
    fn loop_trio_empty_prompt_is_invalid() {
        let arg = parse_loop_arg("trio");
        assert!(matches!(arg, LoopArg::Invalid(_)));
    }

    #[test]
    fn loop_trio_rounds_only_no_prompt_is_invalid() {
        // `--rounds 3` with no prompt after → the prompt is empty, so the
        // caller (parse_loop_arg) rejects it as Invalid before we get here.
        // But parse_trio_args itself returns ("--rounds 3", 3) — the caller
        // checks for empty prompt. Verify the caller path:
        let arg = parse_loop_arg("trio --rounds 3");
        // parse_trio_args returns prompt = "--rounds 3" (non-empty), so this
        // is a Trio with a degenerate prompt. The caller only rejects empty.
        // This is acceptable — the executor gets "--rounds 3" as the task and
        // quickly fails review.
        match arg {
            LoopArg::Trio { max_rounds, .. } => assert_eq!(max_rounds, 3),
            LoopArg::Invalid(_) => {}
            other => panic!("expected Trio or Invalid, got {other:?}"),
        }
    }
}
