use super::*;


/// A minimal agentic coding tool. Works with any OpenAI-compatible endpoint
/// (OpenRouter, pipenetwork.ai, Ollama, llama.cpp, vLLM) or the native
/// Anthropic API.
#[derive(Parser, Debug)]
#[command(name = "hi", version, about)]
pub struct Cli {
    /// Named profile from the config file.
    #[arg(short = 'p', long)]
    pub profile: Option<String>,

    /// Backend wire format.
    #[arg(long, value_enum)]
    pub provider: Option<ProviderName>,

    /// Model id, e.g. "claude-sonnet-4-20250514" or "qwen2.5-coder".
    #[arg(short = 'm', long)]
    pub model: Option<String>,

    /// Override the endpoint base URL.
    #[arg(long)]
    pub base_url: Option<String>,

    /// Override the Pipe MCP endpoint URL used for model discovery.
    #[arg(long)]
    pub mcp_url: Option<String>,

    /// API key (otherwise read from env; see --help).
    #[arg(long)]
    pub api_key: Option<String>,

    /// Fallback profile to try if the primary returns nothing or errors
    /// (repeatable). Also settable per-profile via `fallback = [...]`.
    #[arg(long, value_name = "PROFILE")]
    pub fallback: Vec<String>,

    /// Max output tokens per response.
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (e.g. for varying best-of-N candidates).
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Enable reasoning with this thinking-token budget (Anthropic).
    #[arg(long, value_name = "BUDGET")]
    pub thinking: Option<u32>,

    /// Reasoning effort for OpenAI-compatible endpoints that support it
    /// (minimal, low, medium, high, xhigh). Sent as `reasoning_effort`.
    #[arg(long, value_enum)]
    pub reasoning_effort: Option<CliReasoningEffort>,

    /// Tool calling mode: auto, required, chat-only, or read-only.
    #[arg(long, value_enum)]
    pub tool_mode: Option<CliToolMode>,

    /// Provider compatibility policy: auto retries simpler request shapes; strict sends one shape.
    #[arg(long, value_enum)]
    pub compat: Option<CliCompatMode>,

    /// Path to a config file (default: ./hi.toml or ~/.config/hi/config.toml).
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Print the resolved configuration (provider, model, base URL, etc.) and exit.
    #[arg(long)]
    pub show_config: bool,

    /// Resume the most recent session.
    #[arg(short = 'c', long = "continue")]
    pub cont: bool,

    /// Resume a specific session by id.
    #[arg(long)]
    pub resume: Option<String>,

    /// Don't save this session to disk.
    #[arg(long)]
    pub no_save: bool,

    /// Sync this session to an ipop API endpoint for cross-machine resume.
    /// Reads `HI_SYNC_BASE_URL` and `HI_SYNC_API_KEY` (or uses the provider's
    /// base_url/api_key if those aren't set). Implied by `--sync-session-id`.
    #[arg(long)]
    pub sync: bool,

    /// Explicit session id for sync (otherwise derived from the local session
    /// file stem). Useful when a daemon re-registers an existing session.
    #[arg(long, value_name = "ID")]
    pub sync_session_id: Option<String>,

    /// Run as a subagent (used internally by the `delegate` write-subagent): the
    /// agent won't be offered `explore`/`delegate` (depth ≤ 1) and never saves a
    /// session. Not intended for direct use.
    #[arg(long, hide = true)]
    pub subagent: bool,

    /// Exact session file to create or resume (used internally by the
    /// `/dashboard` fleet: the parent owns the path, so a child running in a
    /// worktree appends to the parent project's session rather than creating
    /// one in the worktree's bucket). Not intended for direct use.
    #[arg(long, hide = true)]
    pub session_file: Option<std::path::PathBuf>,

    /// Set a long-horizon goal (planner-decomposed) before the one-shot turn
    /// runs — used by `/dashboard` goal-driven rows. Ignored when the session
    /// already carries a goal (later fleet turns must not re-plan).
    #[arg(long, hide = true)]
    pub goal: Option<String>,

    /// List saved sessions, then exit.
    #[arg(long)]
    pub list_sessions: bool,

    /// Run the headless `/loop` daemon: keep this project's loops firing (and
    /// auto-fixing) in the background, without the TUI, until Ctrl-C.
    #[arg(long)]
    pub loops_daemon: bool,

    /// Run as a persistent session daemon: hold the agent resident and accept
    /// input from remote clients via ipop. Requires `--sync`. The daemon
    /// long-polls ipop for queued inputs, runs each as a turn, and streams
    /// live events back. Runs until Ctrl-C or the session is ended.
    #[arg(long)]
    pub daemon: bool,

    /// Attach to a running session as a read-only viewer + input sender. Fetches
    /// the session history from ipop, subscribes to the live event stream, and
    /// forwards typed prompts to the hosting daemon. Requires `--sync`.
    #[arg(long, value_name = "SESSION_ID")]
    pub attach: Option<String>,

    /// When used with `--attach`, take over the session in this process instead of
    /// just viewing it. Fetches the durable record history from ipop, reconstructs
    /// the conversation, and boots a local agent that continues from there. Useful
    /// when the daemon is down and you want to keep working.
    #[arg(long)]
    pub resume_local: bool,

    /// The per-session input token for submitting prompts to a token-protected
    /// session via `--attach`. Normally obtained from the daemon's output or a
    /// local file the daemon writes.
    #[arg(long, value_name = "TOKEN")]
    pub input_token: Option<String>,

    /// Use the plain line-based REPL instead of the full-screen TUI.
    #[arg(long)]
    pub plain: bool,

    /// Disable auto-compaction (reclaim context when the window fills).
    #[arg(long)]
    pub no_auto_compact: bool,

    /// Disable the end-of-turn finalization call that writes a structured recap
    /// after a turn changes files (saves one model call per such turn).
    #[arg(long)]
    pub no_finalize: bool,

    /// Disable auto-memory: at the end of an interactive session, distill durable
    /// lessons into `.hi/memory.md` (loaded as context next session).
    #[arg(long)]
    pub no_memory: bool,

    /// Confirm each file edit before applying it. Shows a diff preview and
    /// asks for approval (y/n) before write/edit/multi_edit/apply_patch.
    #[arg(long)]
    pub confirm_edits: bool,

    /// Compaction strategy: hybrid (default), full, or elide.
    #[arg(long, value_name = "KIND")]
    pub compaction: Option<String>,

    /// Verification command. Repeat to replace the automatic pipeline with
    /// multiple ordered stages.
    #[arg(long, value_name = "CMD", action = clap::ArgAction::Append, conflicts_with = "no_verify")]
    pub verify: Vec<String>,

    /// Disable deterministic verification. Mutating work remains unverified.
    #[arg(long, conflicts_with = "verify")]
    pub no_verify: bool,

    /// Permit one-shot unverified mutation to exit successfully.
    #[arg(long)]
    pub allow_unverified: bool,

    /// Permit mutation without a checkpoint even when edit confirmations are enabled.
    #[arg(long)]
    pub allow_no_checkpoint: bool,

    /// Repair/check cycles after the initial verification check.
    #[arg(long)]
    pub max_verify_repairs: Option<u32>,

    /// Independent-review policy.
    #[arg(long, value_enum)]
    pub review: Option<CliReviewPolicy>,

    /// Language-server policy.
    #[arg(long, value_enum)]
    pub lsp: Option<CliLspMode>,

    /// Tool advertisement policy.
    #[arg(long, value_enum)]
    pub tool_set: Option<CliToolSet>,

    /// Safety cap on model calls per turn (stops runaway tool loops).
    #[arg(long)]
    pub max_steps: Option<u32>,

    /// Safety cap on tool executions per turn, independent of model calls.
    #[arg(long, value_name = "N")]
    pub max_tool_calls: Option<u32>,

    /// Execute coding turns remotely through the authenticated Pipe RSI service.
    #[arg(long, conflicts_with = "no_rsi")]
    pub rsi: bool,

    /// Disable remote RSI execution for this process.
    #[arg(long, conflicts_with = "rsi")]
    pub no_rsi: bool,

    /// Run under the trusted RSI worker contract.
    #[arg(
        long,
        hide = true,
        requires_all = ["rsi_trace_dir", "rsi_max_bytes", "rsi_runtime_descriptor"]
    )]
    pub rsi_managed: bool,

    /// Exact empty directory in which managed evidence must be written.
    #[arg(long, hide = true, value_name = "PATH")]
    pub rsi_trace_dir: Option<PathBuf>,

    /// Hard managed evidence capacity.
    #[arg(long, hide = true, value_name = "BYTES")]
    pub rsi_max_bytes: Option<u64>,

    /// Worker-owned Unix socket for managed inference.
    #[arg(long, hide = true, value_name = "PATH", requires = "rsi_managed")]
    pub api_unix_socket: Option<PathBuf>,

    /// Worker-owned bounded prior-conversation document for managed RSI.
    #[arg(long, hide = true, value_name = "PATH", requires = "rsi_managed")]
    pub rsi_context_json: Option<PathBuf>,

    /// Worker-owned, bounded execution descriptor derived from the verified
    /// candidate manifest and lease.
    #[arg(long, hide = true, value_name = "PATH", requires = "rsi_managed")]
    pub rsi_runtime_descriptor: Option<PathBuf>,

    /// Run N candidate attempts in isolated git worktrees and keep the first
    /// that passes the resolved verification pipeline. Requires a prompt.
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub best_of: u32,

    /// Write a JSON usage/outcome report to this path (for eval/automation).
    #[arg(long, value_name = "PATH")]
    pub report: Option<PathBuf>,

    /// Quiet: print only the assistant's text (no tool chatter or usage line).
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// Offline detector eval of the skeptic reviewer: read a JSON
    /// `{objective, sub_goal, diff}` from stdin, run the real skeptic review, print
    /// `{objected, objections}` JSON, and exit. Reviewer = `HI_SKEPTIC_MODEL` or
    /// `--model`. Used by `hi-eval --skeptic-detector`.
    #[arg(long, hide = true)]
    pub skeptic_review: bool,

    /// One-shot prompt. If omitted, starts an interactive session.
    pub prompt: Option<String>,
}

/// YOLO continues silently without checkpoints (telemetry still records it).
/// Opt-in edit confirmation is strict unless this override is supplied.
pub(crate) fn permits_missing_checkpoint(cli: &Cli) -> bool {
    cli.allow_no_checkpoint || !cli.confirm_edits
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ProviderName {
    /// OpenAI-compatible Chat Completions (default base URL: OpenRouter).
    Openai,
    /// Native Anthropic Messages API.
    Anthropic,
    /// pipenetwork.ai — OpenAI-compatible coding-agent endpoint.
    Pipenetwork,
    /// A local Ollama server (OpenAI-compatible).
    Ollama,
    /// xAI (Grok) — OpenAI-compatible. Also the provider for grok.com
    /// subscription login; see `default_base_url` for the endpoint split.
    Xai,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum CliToolMode {
    Auto,
    Required,
    ChatOnly,
    ReadOnly,
}

impl From<CliToolMode> for ToolMode {
    fn from(value: CliToolMode) -> Self {
        match value {
            CliToolMode::Auto => Self::Auto,
            CliToolMode::Required => Self::Required,
            CliToolMode::ChatOnly => Self::ChatOnly,
            CliToolMode::ReadOnly => Self::ReadOnly,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum CliCompatMode {
    Auto,
    Strict,
}

impl From<CliCompatMode> for CompatMode {
    fn from(value: CliCompatMode) -> Self {
        match value {
            CliCompatMode::Auto => Self::Auto,
            CliCompatMode::Strict => Self::Strict,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum CliReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum CliReviewPolicy {
    Risk,
    Always,
    Off,
}

impl From<CliReviewPolicy> for ReviewPolicy {
    fn from(value: CliReviewPolicy) -> Self {
        match value {
            CliReviewPolicy::Risk => Self::Risk,
            CliReviewPolicy::Always => Self::Always,
            CliReviewPolicy::Off => Self::Off,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum CliLspMode {
    Auto,
    On,
    Off,
}

impl From<CliLspMode> for LspMode {
    fn from(value: CliLspMode) -> Self {
        match value {
            CliLspMode::Auto => Self::Auto,
            CliLspMode::On => Self::On,
            CliLspMode::Off => Self::Off,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum CliToolSet {
    Dynamic,
    Minimal,
    Full,
}

impl From<CliToolSet> for ToolSet {
    fn from(value: CliToolSet) -> Self {
        match value {
            CliToolSet::Dynamic => Self::Dynamic,
            CliToolSet::Minimal => Self::Minimal,
            CliToolSet::Full => Self::Full,
        }
    }
}

impl From<CliReasoningEffort> for ReasoningEffort {
    fn from(value: CliReasoningEffort) -> Self {
        match value {
            CliReasoningEffort::Minimal => Self::Minimal,
            CliReasoningEffort::Low => Self::Low,
            CliReasoningEffort::Medium => Self::Medium,
            CliReasoningEffort::High => Self::High,
            CliReasoningEffort::Xhigh => Self::Xhigh,
        }
    }
}

impl ProviderName {
    /// True if this provider speaks the native Anthropic wire format.
    pub fn is_anthropic(self) -> bool {
        matches!(self, ProviderName::Anthropic)
    }

    pub(crate) fn default_base_url(self) -> &'static str {
        match self {
            ProviderName::Openai => "https://openrouter.ai/api/v1",
            ProviderName::Anthropic => "https://api.anthropic.com",
            ProviderName::Pipenetwork => "https://api.pipenetwork.ai/v1",
            ProviderName::Ollama => "http://localhost:11434/v1",
            // The metered API endpoint, used with an XAI_API_KEY. A grok.com
            // subscription login routes to a different host instead — xAI's own
            // CLI keeps these strictly separate, so the OAuth path overrides this.
            ProviderName::Xai => "https://api.x.ai/v1",
        }
    }

    pub(crate) fn default_mcp_url(self) -> Option<&'static str> {
        match self {
            ProviderName::Pipenetwork => Some(hi_ai::PIPE_MCP_DEFAULT_URL),
            _ => None,
        }
    }

    /// A sensible default model for presets that have an obvious one.
    pub(crate) fn default_model(self) -> Option<&'static str> {
        match self {
            ProviderName::Pipenetwork => Some("ipop/coder-balanced"),
            ProviderName::Anthropic => Some("claude-sonnet-4-20250514"),
            // grok-4.3 speaks Chat Completions, the wire format this client uses.
            // grok-4.5 is newer but pi routes it through the Responses API, so it
            // isn't a safe default here until we've confirmed it on /chat/completions.
            ProviderName::Xai => Some("grok-4.3"),
            _ => None,
        }
    }

    /// The lowercase name used in config files / `--provider`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ProviderName::Openai => "openai",
            ProviderName::Anthropic => "anthropic",
            ProviderName::Pipenetwork => "pipenetwork",
            ProviderName::Ollama => "ollama",
            ProviderName::Xai => "xai",
        }
    }

    /// Env vars checked for the API key, in order.
    pub fn key_envs(self) -> &'static [&'static str] {
        match self {
            ProviderName::Anthropic => &["HI_API_KEY", "ANTHROPIC_API_KEY"],
            ProviderName::Pipenetwork => &["PIPENETWORK_API_KEY", "HI_API_KEY", "OPENAI_API_KEY"],
            ProviderName::Ollama => &["HI_API_KEY", "OLLAMA_API_KEY"],
            ProviderName::Openai => &["HI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"],
            ProviderName::Xai => &["XAI_API_KEY", "HI_API_KEY"],
        }
    }
}

impl std::str::FromStr for ProviderName {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "openai" => Ok(Self::Openai),
            "anthropic" => Ok(Self::Anthropic),
            "pipenetwork" => Ok(Self::Pipenetwork),
            "ollama" => Ok(Self::Ollama),
            "xai" => Ok(Self::Xai),
            other => Err(format!(
                "unknown provider '{other}' (expected: openai, anthropic, pipenetwork, ollama, xai)"
            )),
        }
    }
}
