//! CLI parsing, config-file profiles, and resolution into effective settings.
//!
//! Precedence, highest first: explicit CLI flags → selected profile → env vars
//! → built-in defaults. Profiles let a user keep several models on hand
//! (e.g. a cloud Anthropic profile and a local Ollama profile) and use one with
//! `-p <name>`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use hi_agent::{LspMode, ReviewPolicy, ToolSet, VerificationMode, VerifyStage};
use hi_ai::{CompatMode, ReasoningEffort, ToolMode};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_TOKENS: u32 = 8192;
const PIPENETWORK_DEFAULT_MAX_TOKENS: u32 = DEFAULT_MAX_TOKENS;
const LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS: u32 = 2048;

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
        }
    }

    /// Env vars checked for the API key, in order.
    pub fn key_envs(self) -> &'static [&'static str] {
        match self {
            ProviderName::Anthropic => &["HI_API_KEY", "ANTHROPIC_API_KEY"],
            ProviderName::Pipenetwork => &["PIPENETWORK_API_KEY", "HI_API_KEY", "OPENAI_API_KEY"],
            ProviderName::Ollama => &["HI_API_KEY", "OLLAMA_API_KEY"],
            ProviderName::Openai => &["HI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"],
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
            other => Err(format!(
                "unknown provider '{other}' (expected: openai, anthropic, pipenetwork, ollama)"
            )),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    pub default_profile: Option<String>,
    #[serde(default)]
    pub moa: hi_ai::MoaConfig,
    #[serde(default)]
    pub profiles: HashMap<String, Profile>,
    #[serde(default)]
    pub sync: Option<SyncSection>,
}

/// The `[sync]` section in `hi.toml` — configures cross-machine session sync.
/// All fields optional; unset fields fall back to env vars or the provider's
/// credentials.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SyncSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub machine_id: Option<String>,
    /// Persisted sync policy. Missing values migrate from legacy `enabled`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<crate::sync_store::SyncMode>,
    /// When true, sync is enabled by default (no need for `--sync` on the CLI).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub enabled: bool,
}

impl serde::Serialize for Config {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Config", 3)?;
        if let Some(v) = &self.default_profile {
            s.serialize_field("default_profile", v)?;
        }
        if self.moa != hi_ai::MoaConfig::default() {
            s.serialize_field("moa", &self.moa)?;
        }
        if !self.profiles.is_empty() {
            // BTreeMap serializes as a sorted map → stable, alphabetical output.
            let sorted: BTreeMap<&String, &Profile> = self.profiles.iter().collect();
            s.serialize_field("profiles", &sorted)?;
        }
        if let Some(sync) = &self.sync {
            s.serialize_field("sync", sync)?;
        }
        s.end()
    }
}

// Serialized with `skip_serializing_if` on every field so a saved config omits
// unset keys instead of filling with `model = ""` lines. Keep the attribute on
// each new field: a field missing it is fine, but a field missing from
// serialization entirely (as with the old hand-written `Serialize` impl) is
// silently deleted from the user's config file on every save.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Profile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderName>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// MCP endpoint used for metadata discovery, when supported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_url: Option<String>,
    /// A literal API key (written by the setup wizard).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Name of an env var holding the API key for this profile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
    /// Reasoning effort (`reasoning_effort`) for OpenAI-compatible endpoints
    /// that support it. TOML values: minimal/low/medium/high/xhigh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_mode: Option<ToolMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<CompatMode>,
    /// Verifier-gated skill auto-curation: after a verified turn, distill a
    /// reusable technique into a learned skill. Defaults to off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curate_skills: Option<bool>,
    /// Advertise the read-only `explore` subagent tool. On by default; set to
    /// false to disable (e.g. for a very small local model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explore_subagents: Option<bool>,
    /// Advertise the write-capable `delegate` subagent tool. Off by default (the
    /// riskier tier); set to true to enable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_subagents: Option<bool>,
    /// Model id that decomposes a `/goal <objective>` into sub-goals. Defaults to
    /// `pipe/glm-5.2-fast` on the pipenetwork profile; `None` disables planning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_model: Option<String>,
    /// Model id for the `/goal team` skeptic gate (reviews a turn before it
    /// advances a sub-goal). `None` (default) disables the gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skeptic_model: Option<String>,
    /// Other profile names to fall back to, in order, when this one returns
    /// nothing or errors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<Vec<String>>,
}

/// Fully-resolved settings used to build a provider and run the agent.
#[derive(Debug)]
pub struct Settings {
    pub provider: ProviderName,
    pub model: String,
    pub base_url: String,
    pub mcp_url: Option<String>,
    pub api_key: String,
    pub max_tokens: u32,
    pub max_tokens_explicit: bool,
    pub thinking_budget: Option<u32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub tool_mode: ToolMode,
    pub compat: CompatMode,
    pub curate_skills: bool,
    pub explore_subagents: bool,
    pub write_subagents: bool,
    pub planner_model: Option<String>,
    pub skeptic_model: Option<String>,
    pub moa: hi_ai::MoaConfig,
}

/// Resolved project-quality settings. Precedence is CLI, `.hi/config.toml`,
/// then built-in automatic detection/defaults.
#[derive(Clone, Debug)]
pub struct QualitySettings {
    pub verification: VerificationMode,
    pub max_verify_repairs: u32,
    pub review: ReviewPolicy,
    pub lsp_mode: LspMode,
    pub tool_set: ToolSet,
    pub context_exclusions: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ProjectConfig {
    #[serde(default)]
    quality: ProjectQuality,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ProjectVerificationMode {
    Auto,
    Explicit,
    Disabled,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ProjectQuality {
    #[serde(default, alias = "verification_mode")]
    verification: Option<ProjectVerificationMode>,
    /// Ordered commands used by `verification = "explicit"`. `verify` is an
    /// accepted alias for early 0.2 preview files.
    #[serde(default, alias = "verify")]
    stages: Vec<String>,
    #[serde(default)]
    max_verify_repairs: Option<u32>,
    #[serde(default)]
    review: Option<ReviewPolicy>,
    #[serde(default, alias = "lsp_mode")]
    lsp: Option<LspMode>,
    #[serde(default)]
    tool_set: Option<ToolSet>,
    #[serde(default)]
    context_exclusions: Vec<String>,
}

/// Load and resolve `.hi/config.toml` quality policy for `root`.
pub fn resolve_quality(cli: &Cli, root: &Path) -> Result<QualitySettings> {
    let path = root.join(".hi/config.toml");
    let project = if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading project config {}", path.display()))?;
        toml::from_str::<ProjectConfig>(&text)
            .with_context(|| format!("parsing project config {}", path.display()))?
    } else {
        ProjectConfig::default()
    };
    let quality = project.quality;

    let project_verification = match quality.verification {
        Some(ProjectVerificationMode::Disabled) => {
            if !quality.stages.is_empty() {
                bail!("[quality] cannot combine verification = \"disabled\" with stages");
            }
            VerificationMode::Disabled
        }
        Some(ProjectVerificationMode::Explicit) => {
            if quality.stages.is_empty() {
                bail!("[quality] verification = \"explicit\" requires at least one stage");
            }
            VerificationMode::Explicit(quality_stages(&quality.stages)?)
        }
        Some(ProjectVerificationMode::Auto) => {
            if !quality.stages.is_empty() {
                bail!("[quality] cannot combine verification = \"auto\" with stages");
            }
            VerificationMode::Auto
        }
        None if !quality.stages.is_empty() => {
            VerificationMode::Explicit(quality_stages(&quality.stages)?)
        }
        None => VerificationMode::Auto,
    };

    let verification = if cli.no_verify {
        VerificationMode::Disabled
    } else if !cli.verify.is_empty() {
        VerificationMode::Explicit(quality_stages(&cli.verify)?)
    } else {
        project_verification
    };

    Ok(QualitySettings {
        verification,
        max_verify_repairs: cli
            .max_verify_repairs
            .or(quality.max_verify_repairs)
            .unwrap_or(2),
        review: cli
            .review
            .map(ReviewPolicy::from)
            .or(quality.review)
            .unwrap_or_default(),
        lsp_mode: cli
            .lsp
            .map(LspMode::from)
            .or(quality.lsp)
            .unwrap_or_default(),
        tool_set: cli
            .tool_set
            .map(ToolSet::from)
            .or(quality.tool_set)
            .unwrap_or_default(),
        context_exclusions: quality.context_exclusions,
    })
}

fn quality_stages(commands: &[String]) -> Result<Vec<VerifyStage>> {
    if let Some((index, _)) = commands
        .iter()
        .enumerate()
        .find(|(_, command)| command.trim().is_empty())
    {
        bail!("verification stage {} must not be empty", index + 1);
    }
    Ok(commands
        .iter()
        .enumerate()
        .map(|(index, command)| {
            let name = if commands.len() == 1 {
                "verify".to_string()
            } else {
                format!("verify_{}", index + 1)
            };
            VerifyStage::new(name, command.trim().to_string())
        })
        .collect())
}

pub fn load_config(explicit: Option<&Path>) -> Result<Config> {
    if let Some(path) = explicit {
        return read_config(path);
    }

    let mut config = default_config_path()
        .filter(|path| path.exists())
        .map(|path| read_config(&path))
        .transpose()?
        .unwrap_or_default();

    let local_path = local_config_path();
    if local_path.exists() {
        let local = read_config(&local_path)?;
        merge_config(&mut config, local);
    }

    config.moa.validate()?;
    Ok(config)
}

fn read_config(path: &Path) -> Result<Config> {
    let mut config = read_config_file(path)?;
    config
        .moa
        .validate()
        .with_context(|| format!("validating MoA config {}", path.display()))?;
    migrate_api_key_env_to_literal(&mut config, path);
    Ok(config)
}

/// Parse a single config file as-is: no validation, no key migration. Used by
/// the read-modify-write save path, which must reproduce the file's own
/// contents faithfully rather than the session's merged/migrated view.
fn read_config_file(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config {}", path.display()))?;
    toml::from_str::<Config>(&text).with_context(|| format!("parsing config {}", path.display()))
}

fn merge_config(base: &mut Config, overlay: Config) {
    if overlay.default_profile.is_some() {
        base.default_profile = overlay.default_profile;
    }
    if overlay.moa != hi_ai::MoaConfig::default() {
        base.moa = overlay.moa;
    }
    base.profiles.extend(overlay.profiles);
}

/// Repair profiles whose `api_key_env` holds a literal key instead of an env
/// var name — a bug in the setup wizard before the env-var-reference check was
/// added. The wizard stored a pasted key as `api_key_env` whenever it looked
/// like an env var name (all caps + digits + underscores), so on the next run
/// `resolve_api_key_for` tried to read an env var with that name and failed
/// with "env var … (from profile) is not set".
///
/// Fix: if `api_key_env` is set but no env var with that name exists in the
/// environment, AND the value does NOT look like an env var name (i.e. it's a
/// real key that was misclassified), move it to `api_key` (literal). If the
/// value *does* look like an env var name and the env var isn't set, leave it
/// alone — that's a legitimate but unfulfilled env var reference (the user
/// needs to set the env var), not a misplaced literal key. Moving an env var
/// name like "HI_API_KEY" to `api_key` would authenticate with the literal
/// string "HI_API_KEY" and get a 401.
fn migrate_api_key_env_to_literal(config: &mut Config, path: &Path) {
    let mut changed = false;
    // Set when a repair copies a live secret out of the environment into
    // `api_key` (below). We keep that repair in memory so the session works, but
    // must NOT persist it: writing the resolved secret to disk would leak it into
    // the file — including a project-local `hi.toml` that is routinely committed
    // to git — turning what was an env-var reference into a checked-in credential.
    let mut materialized_env_secret = false;
    for profile in config.profiles.values_mut() {
        // First, repair a bad migration from an earlier version of this fix:
        // the previous migration moved an env var *name* (like "HI_API_KEY")
        // into `api_key` when the env var wasn't set, causing 401s. If `api_key`
        // looks like an env var name and that env var is set, replace the value
        // with the env var's contents. If the env var isn't set, convert it back
        // to an `api_key_env` reference (the user intended an env var, not a
        // literal — they need to set it).
        if let Some(key) = profile.api_key.clone()
            && looks_like_env_var_name(&key)
        {
            if let Ok(val) = std::env::var(&key)
                && !val.is_empty()
            {
                // The env var is set — use its value as the literal key
                // in-memory only (do not persist the resolved secret to disk).
                profile.api_key = Some(val);
                changed = true;
                materialized_env_secret = true;
            } else {
                // Env var not set — this is an env var reference, not a key.
                // Move it back to api_key_env so resolve_api_key_for gives the
                // right error message ("env var … is not set") instead of a 401.
                profile.api_key = None;
                profile.api_key_env = Some(key);
                changed = true;
            }
        }

        let Some(env_name) = profile.api_key_env.clone() else {
            continue;
        };
        // If the env var is actually set, this is a legitimate reference — leave it.
        if std::env::var(&env_name).is_ok_and(|v| !v.is_empty()) {
            continue;
        }
        // If the value looks like an env var name, it could be a legitimate
        // (but unset) env var reference that the user intentionally configured.
        // BUT: the old setup wizard (save_config) always wrote api_key_env =
        // key_envs().first() (e.g. "HI_API_KEY") regardless of what the user
        // entered — it never stored the actual key. That pattern is: api_key_env
        // is one of the provider's standard key env names, the env var isn't set,
        // and there's no literal api_key. In that case, drop the bogus reference
        // so resolve falls through to the env-var candidates and the onboarding
        // error, prompting the user to re-enter their key (the new wizard stores
        // it as a literal api_key).
        if looks_like_env_var_name(&env_name) {
            let provider = profile.provider.unwrap_or(ProviderName::Openai);
            let is_standard = provider.key_envs().iter().any(|n| *n == env_name);
            let has_literal_key = profile.api_key.is_some();
            if is_standard && !has_literal_key {
                // Old buggy save_config output — drop it.
                profile.api_key_env = None;
                changed = true;
            }
            continue;
        }
        // The value doesn't look like an env var name and the env var isn't set:
        // it's a literal key that was misclassified by the old wizard heuristic.
        // Move it to `api_key` and clear `api_key_env`.
        profile.api_key_env = None;
        if profile.api_key.is_none() {
            profile.api_key = Some(env_name);
        }
        changed = true;
    }
    if changed && !materialized_env_secret {
        // Best-effort rewrite; if it fails we've still repaired the in-memory
        // config so this run works, just not the next one. Route through
        // `save_config_to` so the file keeps 0600 (a bare `fs::write` would drop
        // permissions, leaving keys world-readable). Skipped when the repair
        // materialized a live env secret — that stays in memory only (see above).
        let _ = save_config_to(config, path);
    }
}

fn local_config_path() -> PathBuf {
    PathBuf::from("hi.toml")
}

/// Apply precedence to produce the effective [`Settings`].
pub fn resolve(cli: &Cli, config: &Config) -> Result<Settings> {
    config.moa.validate()?;
    let profile = match cli.profile.as_ref().or(config.default_profile.as_ref()) {
        Some(name) => Some(
            config
                .profiles
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow!("profile '{name}' not found in config"))?,
        ),
        None => None,
    };
    let profile = profile.as_ref();

    let provider_explicit = cli.provider.is_some() || profile.is_some_and(|p| p.provider.is_some());
    let mut provider = cli
        .provider
        .or(profile.and_then(|p| p.provider))
        .unwrap_or(ProviderName::Openai);

    let mut model = cli
        .model
        .clone()
        .or_else(|| profile.and_then(|p| p.model.clone()))
        .or_else(|| std::env::var("HI_MODEL").ok())
        .or_else(|| provider.default_model().map(String::from));

    // Bare run with nothing configured: infer a provider+model from the
    // environment so `hi` "just works" when a key is present.
    if model.is_none()
        && !provider_explicit
        && let Some((auto_provider, auto_model)) = auto_select()
    {
        provider = auto_provider;
        model = Some(auto_model);
    }
    let model = model.ok_or_else(|| anyhow!("{ONBOARDING}"))?;

    let base_url = cli
        .base_url
        .clone()
        .or_else(|| profile.and_then(|p| p.base_url.clone()))
        .or_else(|| std::env::var("HI_BASE_URL").ok())
        .unwrap_or_else(|| provider.default_base_url().to_string());

    let mcp_url = cli
        .mcp_url
        .clone()
        .or_else(|| profile.and_then(|p| p.mcp_url.clone()))
        .or_else(|| std::env::var("HI_MCP_URL").ok())
        .or_else(|| provider.default_mcp_url().map(String::from));

    let api_key = resolve_api_key(cli, profile, provider)?;

    let profile_max_tokens = profile.and_then(|p| p.max_tokens);
    let max_tokens = configured_max_tokens(provider, cli.max_tokens, profile_max_tokens);
    let max_tokens_explicit = max_tokens_is_explicit(provider, cli.max_tokens, profile_max_tokens);

    let thinking_budget = cli.thinking.or(profile.and_then(|p| p.thinking_budget));
    let reasoning_effort = cli
        .reasoning_effort
        .map(ReasoningEffort::from)
        .or_else(|| profile.and_then(|p| p.reasoning_effort));
    let tool_mode = cli
        .tool_mode
        .map(ToolMode::from)
        .or_else(|| profile.and_then(|p| p.tool_mode))
        .unwrap_or_default();
    let compat = cli
        .compat
        .map(CompatMode::from)
        .or_else(|| profile.and_then(|p| p.compat))
        .unwrap_or_default();
    let curate_skills = curate_skills_default(provider, profile.and_then(|p| p.curate_skills));
    let explore_subagents = explore_subagents_default(profile.and_then(|p| p.explore_subagents));
    let write_subagents = profile.and_then(|p| p.write_subagents).unwrap_or(false);
    let planner_model =
        planner_model_default(provider, profile.and_then(|p| p.planner_model.clone()));
    // Skeptic model: opt-in, no provider default (unlike the planner) — off unless
    // a profile or HI_SKEPTIC_MODEL sets it.
    let skeptic_model = profile.and_then(|p| p.skeptic_model.clone());

    Ok(Settings {
        provider,
        model,
        base_url,
        mcp_url,
        api_key,
        max_tokens,
        max_tokens_explicit,
        thinking_budget,
        reasoning_effort,
        tool_mode,
        compat,
        curate_skills,
        explore_subagents,
        write_subagents,
        planner_model,
        skeptic_model,
        moa: config.moa.clone(),
    })
}

/// Guess a *layered* verification pipeline from marker files in `dir`: a cheap
/// compile/typecheck (and lint, when obviously configured) before tests, so the
/// model gets fast, localizable errors before the slower test stage. Used by
/// automatic verification so the proven verify-loop is zero-config. Empty =
/// unknown project.
#[cfg(test)]
pub fn detect_verify_pipeline(dir: &Path) -> Vec<VerifyStage> {
    hi_agent::detect_verify_pipeline(dir)
}

/// True when nothing is configured — used to trigger the interactive setup
/// wizard on a fresh terminal.
pub fn needs_setup(cli: &Cli, file: &Config) -> bool {
    cli.model.is_none()
        && cli.provider.is_none()
        && cli.profile.is_none()
        && file.default_profile.is_none()
        // Only treat this as a first run when there are no profiles at all. A
        // user who defines profiles but no `default_profile` (they always launch
        // with `-p <name>`) must NOT get the setup wizard on a bare `hi` — its
        // `save_config` blindly overwrites the entire config file with a single
        // hardcoded profile, destroying every existing profile and its API key.
        && file.profiles.is_empty()
        && std::env::var("HI_MODEL").is_err()
        && auto_select().is_none()
}

/// The default config file path to write the wizard's choices to.
pub fn default_config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join("hi").join("config.toml"))
}

/// The path to write config to: an explicit `--config` path, a local `hi.toml`
/// if it exists, or the default global path. Unlike [`config_path`], this
/// returns a path even when the file doesn't exist yet (so we can create it).
pub fn writable_config_path(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    let local = PathBuf::from("hi.toml");
    if local.exists() {
        return Some(local);
    }
    default_config_path()
}

/// Mask an API key (or env var name) for display: first and last four
/// characters with an ellipsis. Char-based, so a key containing multi-byte
/// characters (e.g. pasted with a stray curly quote) can't panic a byte slice.
pub fn mask_key(key: &str) -> String {
    if key.is_empty() {
        return "(none)".to_string();
    }
    let chars: Vec<char> = key.chars().collect();
    if chars.len() > 8 {
        let head: String = chars[..4].iter().collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{head}…{tail}")
    } else {
        "***".to_string()
    }
}

/// Serialize `config` to TOML and write it to `path`, creating parent dirs.
/// Sets 0600 permissions on Unix so API keys in the file aren't world-readable.
pub fn save_config_to(config: &Config, path: &Path) -> Result<()> {
    let toml = toml::to_string_pretty(config)
        .with_context(|| format!("serializing config to {}", path.display()))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating config dir {}", parent.display()))?;
    }
    std::fs::write(path, toml).with_context(|| format!("writing {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// The fields needed to create or edit a profile, collected from the user.
/// Used by both the plain REPL prompts and the TUI form.
#[derive(Clone, Debug)]
pub struct ProfileForm {
    pub name: String,
    pub provider: ProviderName,
    pub api_key: String,
    /// Whether to store the key as a literal (`api_key`) or an env var name
    /// (`api_key_env`). The setup wizard uses env vars for cloud providers;
    /// we match that convention.
    pub store_as_env: bool,
    pub model: String,
    pub base_url: String,
}

impl Default for ProfileForm {
    fn default() -> Self {
        Self {
            name: String::new(),
            provider: ProviderName::Openai,
            api_key: String::new(),
            store_as_env: false,
            model: String::new(),
            base_url: String::new(),
        }
    }
}

impl ProfileForm {
    /// Build a `Profile` from the form fields, leaving unused fields as `None`.
    pub fn to_profile(&self) -> Profile {
        let mut p = Profile {
            provider: Some(self.provider),
            ..Default::default()
        };
        if !self.model.is_empty() {
            p.model = Some(self.model.clone());
        }
        if !self.base_url.is_empty() {
            p.base_url = Some(self.base_url.clone());
        }
        if !self.api_key.is_empty() {
            // The API-key field accepts either a literal key or the *name* of an
            // env var that holds the key. Distinguish by checking the environment:
            // if the input is a plausible env var name AND an env var with that
            // name is actually set, store it as an `api_key_env` reference;
            // otherwise treat it as a literal key. This is unambiguous — a real
            // key will never be the name of a set env var, so a pasted key that
            // happens to be all-caps+digits+underscores is stored correctly
            // instead of being mistaken for an env var name (which would fail at
            // resolve time with "env var … is not set").
            if is_env_var_reference(&self.api_key) {
                p.api_key_env = Some(self.api_key.clone());
            } else {
                p.api_key = Some(self.api_key.clone());
            }
        }
        p
    }

    /// Build a `Profile` for an *edit*: start from the existing profile so the
    /// fields the form doesn't cover (max_tokens, thinking_budget, tool_mode,
    /// compat, fallback, subagent/planner settings, mcp_url, …) survive, and
    /// overwrite only what the form actually edits.
    pub fn apply_to(&self, existing: &Profile) -> Profile {
        let form = self.to_profile();
        Profile {
            provider: form.provider,
            model: form.model,
            base_url: form.base_url,
            api_key: form.api_key,
            api_key_env: form.api_key_env,
            ..existing.clone()
        }
    }

    /// Populate the form from an existing profile (for editing).
    pub fn from_profile(name: &str, p: &Profile) -> Self {
        Self {
            name: name.to_string(),
            provider: p.provider.unwrap_or(ProviderName::Openai),
            api_key: p
                .api_key_env
                .clone()
                .or_else(|| p.api_key.clone())
                .unwrap_or_default(),
            store_as_env: p.api_key_env.is_some(),
            model: p.model.clone().unwrap_or_default(),
            base_url: p.base_url.clone().unwrap_or_default(),
        }
    }
}

// --- Layered profile persistence ------------------------------------------
//
// The session's `Config` is the *merge* of the global config and a local
// `hi.toml`, but saves must never serialize that merged view into one file:
// with a local `hi.toml` present that copied every global profile — API keys
// included — into a project file that's easy to commit, and removing a
// globally-defined profile only masked it locally until the next merge
// resurrected it. Instead, every save is a read-modify-write of exactly the
// file(s) that own the data being changed: the explicit `--config` file when
// one was given (the whole session is that single file), else the layer file
// that defines the profile (local first — it wins the merge), else the
// default writable path for brand-new profiles.

/// Read `path` (or start empty if it doesn't exist), apply `mutate` to that
/// file's own contents, and write it back.
fn rmw_config_file(path: &Path, mutate: impl FnOnce(&mut Config)) -> Result<()> {
    let mut file = if path.exists() {
        read_config_file(path)?
    } else {
        Config::default()
    };
    mutate(&mut file);
    save_config_to(&file, path)
}

/// Existing config layer files, highest merge precedence first (local
/// `hi.toml`, then the global config).
fn layer_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let local = local_config_path();
    if local.exists() {
        out.push(local);
    }
    if let Some(global) = default_config_path().filter(|p| p.exists()) {
        out.push(global);
    }
    out
}

/// The layer files (from `layers`, highest precedence first) that define
/// profile `name`. Files that fail to parse are skipped (the profile can't
/// have come from them).
fn layers_defining(layers: &[PathBuf], name: &str) -> Vec<PathBuf> {
    layers
        .iter()
        .filter(|p| {
            read_config_file(p)
                .map(|c| c.profiles.contains_key(name))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}

/// The highest-precedence layer file that defines profile `name`.
fn owning_path_in(layers: &[PathBuf], name: &str) -> Option<PathBuf> {
    layers_defining(layers, name).into_iter().next()
}

/// Where a change to profile `name` must be written: the explicit `--config`
/// path if given, else the layer file that defines the profile, else (a new
/// profile) the default writable path.
fn profile_save_target(name: &str, explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    owning_path_in(&layer_paths(), name)
        .or_else(|| writable_config_path(None))
        .ok_or_else(|| anyhow!("could not determine config path"))
}

/// Add or replace a profile in the config and save it to disk. Only `name`'s
/// entry in the owning file is touched (see module note above).
pub fn upsert_profile(
    config: &mut Config,
    name: &str,
    profile: Profile,
    explicit: Option<&Path>,
) -> Result<()> {
    validate_profile(&profile)?;
    let target = profile_save_target(name, explicit)?;
    rmw_config_file(&target, |file| {
        file.profiles.insert(name.to_string(), profile.clone());
    })?;
    config.profiles.insert(name.to_string(), profile);
    Ok(())
}

/// Add or replace a profile, select it as the default profile, and save.
pub fn upsert_profile_as_default(
    config: &mut Config,
    name: &str,
    profile: Profile,
    explicit: Option<&Path>,
) -> Result<()> {
    upsert_profile(config, name, profile, explicit)?;
    // `default_profile` must land where the merge can't shadow it: the
    // explicit file, else the highest-precedence layer (a local `hi.toml`
    // overrides the global default_profile whenever it sets one), else the
    // default writable path.
    let target = match explicit {
        Some(path) => path.to_path_buf(),
        None => layer_paths()
            .into_iter()
            .next()
            .or_else(|| writable_config_path(None))
            .ok_or_else(|| anyhow!("could not determine config path"))?,
    };
    rmw_config_file(&target, |file| {
        file.default_profile = Some(name.to_string());
    })?;
    config.default_profile = Some(name.to_string());
    Ok(())
}

/// Update only the selected model on an existing profile and save it to disk.
pub fn set_profile_model(
    config: &mut Config,
    name: &str,
    model: &str,
    explicit: Option<&Path>,
) -> Result<()> {
    let profile = config
        .profiles
        .get_mut(name)
        .ok_or_else(|| anyhow!("profile '{name}' not found in config"))?;
    profile.model = Some(model.to_string());
    validate_profile(profile)?;
    let updated = profile.clone();
    let target = profile_save_target(name, explicit)?;
    rmw_config_file(&target, |file| {
        match file.profiles.get_mut(name) {
            // Touch only the model, preserving whatever else that file says
            // about the profile.
            Some(p) => p.model = Some(model.to_string()),
            // The file doesn't define it (deleted mid-session, or a fresh
            // explicit path): write the full in-memory profile.
            None => {
                file.profiles.insert(name.to_string(), updated.clone());
            }
        }
    })
}

/// Remove a profile from the config and save. Returns `false` if the profile
/// didn't exist (caller may treat that as an error or a no-op). Without an
/// explicit path the profile is removed from *every* layer file that defines
/// it — deleting it from just one file would let the merge resurrect it from
/// the other on the next launch.
pub fn remove_profile(config: &mut Config, name: &str, explicit: Option<&Path>) -> Result<bool> {
    let in_memory = config.profiles.remove(name).is_some();
    let targets: Vec<PathBuf> = match explicit {
        Some(path) => vec![path.to_path_buf()],
        None => layers_defining(&layer_paths(), name),
    };
    if !in_memory && targets.is_empty() {
        return Ok(false);
    }
    for path in &targets {
        rmw_config_file(path, |file| {
            file.profiles.remove(name);
        })?;
    }
    Ok(true)
}

/// Sanity-check a profile before saving. Currently validates that the base URL
/// doesn't include an endpoint path (the provider appends `/chat/completions`
/// and `/models` itself — a common copy-paste mistake is to paste the full
/// endpoint URL, which produces 404s).
fn validate_profile(profile: &Profile) -> Result<()> {
    if let Some(url) = &profile.base_url {
        let trimmed = url.trim_end_matches('/');
        for suffix in ["/chat/completions", "/completions", "/messages"] {
            if trimmed.ends_with(suffix) {
                bail!(
                    "base_url looks like a full endpoint path (ends with '{suffix}'). \
                     The provider appends the endpoint path itself — use just the base, \
                     e.g. 'http://localhost:11434/v1' not 'http://localhost:11434/v1{suffix}'."
                );
            }
        }
    }
    Ok(())
}

/// Does `s` look like an env var *name* (not a value)? Env var names are
/// uppercase ASCII letters, digits, and underscores, must contain at least one
/// underscore (single-word names like `PATH` are rare for API-key vars and we
/// err toward treating short all-caps tokens as literal keys), and aren't
/// absurdly long. This is only a pre-filter — the real decision in
/// `to_profile` also requires the var to be set in the environment.
fn looks_like_env_var_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.contains('_')
        && s.chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
}

/// Does `s` look like an env var *name* and is an env var with that name
/// actually set? Used by the wizards to label the field and pre-set
/// `store_as_env`; `to_profile` makes the final decision the same way.
pub(crate) fn is_env_var_reference(s: &str) -> bool {
    looks_like_env_var_name(s) && std::env::var(s).is_ok_and(|v| !v.is_empty())
}

/// Shown when `hi` is run with nothing configured. Actionable, not terse.
const ONBOARDING: &str = "no model configured. Get started with one of:

  pipenetwork.ai:   PIPENETWORK_API_KEY=...  hi --provider pipenetwork \"...\"
  Local (Ollama):   hi --provider ollama -m qwen2.5-coder \"...\"

Or run `hi` on a real terminal for the interactive setup wizard.
Or set HI_MODEL, or add a profile in ~/.config/hi/config.toml (see README).
Tip: interactive sessions use the full-screen interface by default; pass --plain for the line REPL.";

/// Infer a provider + model from API keys present in the environment.
fn auto_select() -> Option<(ProviderName, String)> {
    let set = |name: &str| std::env::var(name).is_ok_and(|v| !v.is_empty());
    if set("PIPENETWORK_API_KEY") {
        Some((ProviderName::Pipenetwork, "ipop/coder-balanced".into()))
    } else if set("ANTHROPIC_API_KEY") {
        Some((ProviderName::Anthropic, "claude-sonnet-4-20250514".into()))
    } else {
        None
    }
}

fn resolve_api_key(cli: &Cli, profile: Option<&Profile>, provider: ProviderName) -> Result<String> {
    if let Some(key) = &cli.api_key {
        return Ok(key.clone());
    }
    // A launcher (best-of / delegate / fleet child) passes the parent's already
    // resolved key here so the child authenticates with the SAME key the parent
    // used. The child re-resolves config from scratch and would otherwise let a
    // default-profile literal `api_key` shadow the parent's key (e.g. when the
    // parent ran with `--profile alt` or `--api-key`), causing silent auth
    // failures. It's passed in the environment, not argv, so it isn't exposed in
    // `ps` — hence it must win over the profile here rather than being a
    // last-resort candidate like `HI_API_KEY`.
    if let Ok(key) = std::env::var("HI_FORCE_API_KEY")
        && !key.is_empty()
    {
        return Ok(key);
    }
    resolve_api_key_for(profile, provider)
}

/// API key for a profile/provider, independent of CLI flags (used for fallback
/// profiles, whose keys come from their own profile or the environment).
fn resolve_api_key_for(profile: Option<&Profile>, provider: ProviderName) -> Result<String> {
    if let Some(key) = profile.and_then(|p| p.api_key.clone()) {
        return Ok(key);
    }
    if let Some(env_name) = profile.and_then(|p| p.api_key_env.as_ref()) {
        return std::env::var(env_name).map_err(|_| {
            anyhow!(
                "env var {env_name} (from profile) is not set.\n\
                 Fix: either `export {env_name}=your-key` and restart hi, or re-run the\n\
                 setup wizard (`hi` with no config) to store the key directly in the config file."
            )
        });
    }
    let candidates = provider.key_envs();
    for name in candidates {
        if let Ok(value) = std::env::var(name)
            && !value.is_empty()
        {
            return Ok(value);
        }
    }
    // A local Ollama server ignores the key, so don't require one.
    if matches!(provider, ProviderName::Ollama) {
        return Ok("ollama".into());
    }
    let names: Vec<String> = candidates.iter().map(|s| s.to_string()).collect();
    let hint = match names.len() {
        0 => "an API key env var".to_string(),
        1 => names[0].clone(),
        _ => format!(
            "{} or {}",
            names[..names.len() - 1].join(", "),
            names[names.len() - 1]
        ),
    };
    bail!("no API key: pass --api-key or set {hint}");
}

/// The sorted list of configured profile names, for `/provider` (no arg).
pub fn profile_names(config: &Config) -> Vec<String> {
    let mut names: Vec<String> = config.profiles.keys().cloned().collect();
    names.sort();
    names
}

/// The fallback chain (excluding the primary) — `--fallback` flags first, then
/// the selected profile's `fallback` list, deduped. Profiles that don't resolve
/// (missing key/model) are skipped with a warning rather than blocking startup.
pub fn resolve_fallbacks(cli: &Cli, config: &Config) -> Vec<Settings> {
    let primary_name = cli.profile.as_ref().or(config.default_profile.as_ref());

    let mut names: Vec<String> = cli.fallback.clone();
    if let Some(name) = primary_name
        && let Some(profile) = config.profiles.get(name)
        && let Some(list) = &profile.fallback
    {
        names.extend(list.iter().cloned());
    }

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(name) = primary_name {
        seen.insert(name.clone()); // don't fall back to the primary itself
    }

    let mut out = Vec::new();
    for name in names {
        if !seen.insert(name.clone()) {
            continue;
        }
        match resolve_named_profile(config, &name) {
            Ok(settings) => out.push(settings),
            Err(err) => {
                eprintln!("\x1b[33mwarning: skipping fallback profile '{name}': {err}\x1b[0m")
            }
        }
    }
    out
}

/// Resolve a named profile into [`Settings`] from its own fields + environment
/// (no CLI overrides — those belong to the primary). Used both for fallback
/// profiles at startup and for `/provider` changes mid-session.
///
/// If the profile has no `model` and the provider has no default, a placeholder
/// is used. The placeholder is fine for building the provider and listing
/// models, but a turn can't run with it.
pub fn resolve_named_profile(config: &Config, name: &str) -> Result<Settings> {
    config.moa.validate()?;
    let profile = config
        .profiles
        .get(name)
        .ok_or_else(|| anyhow!("profile '{name}' not found in config"))?;

    let provider = profile.provider.unwrap_or(ProviderName::Openai);
    let model = profile
        .model
        .clone()
        .or_else(|| provider.default_model().map(String::from))
        .unwrap_or_else(|| "__model_not_configured__".to_string());
    let base_url = profile
        .base_url
        .clone()
        .unwrap_or_else(|| provider.default_base_url().to_string());
    let mcp_url = profile
        .mcp_url
        .clone()
        .or_else(|| std::env::var("HI_MCP_URL").ok())
        .or_else(|| provider.default_mcp_url().map(String::from));
    let api_key = resolve_api_key_for(Some(profile), provider)?;

    let max_tokens = configured_max_tokens(provider, None, profile.max_tokens);
    let max_tokens_explicit = max_tokens_is_explicit(provider, None, profile.max_tokens);

    Ok(Settings {
        provider,
        model,
        base_url,
        mcp_url,
        api_key,
        max_tokens,
        max_tokens_explicit,
        thinking_budget: profile.thinking_budget,
        reasoning_effort: profile.reasoning_effort,
        tool_mode: profile.tool_mode.unwrap_or_default(),
        compat: profile.compat.unwrap_or_default(),
        curate_skills: curate_skills_default(provider, profile.curate_skills),
        explore_subagents: explore_subagents_default(profile.explore_subagents),
        write_subagents: profile.write_subagents.unwrap_or(false),
        planner_model: planner_model_default(provider, profile.planner_model.clone()),
        skeptic_model: profile.skeptic_model.clone(),
        moa: config.moa.clone(),
    })
}

fn configured_max_tokens(
    provider: ProviderName,
    cli_max_tokens: Option<u32>,
    profile_max_tokens: Option<u32>,
) -> u32 {
    if let Some(value) = cli_max_tokens {
        return value;
    }
    match (provider, profile_max_tokens) {
        // Pipenetwork profiles may carry old wizard defaults. Treat those as
        // implicit so live API limits can size coding-agent turns at runtime;
        // an explicit CLI --max-tokens still wins above.
        (
            ProviderName::Pipenetwork,
            None | Some(DEFAULT_MAX_TOKENS) | Some(LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS),
        ) => PIPENETWORK_DEFAULT_MAX_TOKENS,
        (_, Some(value)) => value,
        (_, None) => DEFAULT_MAX_TOKENS,
    }
}

/// Whether verifier-gated skill auto-curation is on. An explicit `curate_skills`
/// in the profile always wins; otherwise it defaults on for the pipenetwork
/// provider (its coding-agent models are strong enough for the curator to pay
/// off) and off for every other provider.
fn curate_skills_default(provider: ProviderName, profile_value: Option<bool>) -> bool {
    profile_value.unwrap_or(provider == ProviderName::Pipenetwork)
}

/// Whether the read-only `explore` subagent tool is advertised. On by default for
/// every provider (the tool is read-only, depth-capped at 1, and per-session
/// budgeted, so it's safe to offer broadly); a profile can set `explore_subagents
/// = false` to turn it off (e.g. for a very small local model that would misuse it).
fn explore_subagents_default(profile_value: Option<bool>) -> bool {
    profile_value.unwrap_or(true)
}

/// The `/goal` planner model. An explicit `planner_model` in the profile always
/// wins; otherwise it defaults to glm-5.2 on pipenetwork (a strong planner served
/// there) and `None` (no decomposition — a single sub-goal) for every other
/// provider, since the id wouldn't route on their endpoint.
fn planner_model_default(provider: ProviderName, profile_value: Option<String>) -> Option<String> {
    profile_value.or_else(|| {
        (provider == ProviderName::Pipenetwork).then(|| "pipe/glm-5.2-fast".to_string())
    })
}

fn max_tokens_is_explicit(
    provider: ProviderName,
    cli_max_tokens: Option<u32>,
    profile_max_tokens: Option<u32>,
) -> bool {
    if cli_max_tokens.is_some() {
        return true;
    }
    match (provider, profile_max_tokens) {
        (
            ProviderName::Pipenetwork,
            None | Some(DEFAULT_MAX_TOKENS) | Some(LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS),
        ) => false,
        (_, Some(_)) => true,
        (_, None) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Config, DEFAULT_MAX_TOKENS, LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS,
        PIPENETWORK_DEFAULT_MAX_TOKENS, Profile, ProviderName, configured_max_tokens,
        curate_skills_default, detect_verify_pipeline, explore_subagents_default,
        max_tokens_is_explicit, permits_missing_checkpoint, planner_model_default, resolve_quality,
        save_config_to,
    };
    use clap::Parser;
    use hi_agent::{LspMode, ReviewPolicy, ToolSet, VerificationMode};
    use std::sync::atomic::{AtomicU32, Ordering};

    fn temp_dir_with(marker: &str) -> std::path::PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "hi-detect-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        if !marker.is_empty() {
            std::fs::write(dir.join(marker), "").unwrap();
        }
        dir
    }

    #[test]
    fn detects_layered_pipeline_by_marker() {
        // (marker, expected stage commands in order)
        let cases: [(&str, Vec<&str>); 6] = [
            (
                "Cargo.toml",
                vec!["cargo check --quiet", "cargo test --quiet"],
            ),
            ("go.mod", vec!["go build ./...", "go test ./..."]),
            ("pyproject.toml", vec!["pytest -q"]),
            ("package.json", vec!["npm test --silent"]),
            ("Makefile", vec!["make test"]),
            ("", vec![]),
        ];
        for (marker, expected) in cases {
            let dir = temp_dir_with(marker);
            let got: Vec<String> = detect_verify_pipeline(&dir)
                .into_iter()
                .map(|s| s.command)
                .collect();
            assert_eq!(got, expected, "marker={marker:?}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn quality_defaults_to_automatic_safe_policy() {
        let dir = temp_dir_with("");
        let cli = super::Cli::try_parse_from(["hi"]).unwrap();
        let quality = resolve_quality(&cli, &dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(quality.verification, VerificationMode::Auto);
        assert_eq!(quality.max_verify_repairs, 2);
        assert_eq!(quality.review, ReviewPolicy::Risk);
        assert_eq!(quality.lsp_mode, LspMode::Auto);
        assert_eq!(quality.tool_set, ToolSet::Dynamic);
        assert!(!cli.allow_no_checkpoint);
        assert!(permits_missing_checkpoint(&cli));
    }

    #[test]
    fn checkpoint_policy_is_yolo_unless_edit_confirmation_is_strict() {
        let default = super::Cli::try_parse_from(["hi"]).unwrap();
        assert!(permits_missing_checkpoint(&default));

        let strict = super::Cli::try_parse_from(["hi", "--confirm-edits"]).unwrap();
        assert!(!permits_missing_checkpoint(&strict));

        let override_cli =
            super::Cli::try_parse_from(["hi", "--confirm-edits", "--allow-no-checkpoint"]).unwrap();
        assert!(permits_missing_checkpoint(&override_cli));
    }

    #[test]
    fn cli_quality_overrides_project_config_and_verify_is_repeatable() {
        let dir = temp_dir_with("");
        std::fs::create_dir_all(dir.join(".hi")).unwrap();
        std::fs::write(
            dir.join(".hi/config.toml"),
            r#"[quality]
verification = "disabled"
max_verify_repairs = 7
review = "off"
lsp = "off"
tool_set = "full"
context_exclusions = ["generated/**"]
"#,
        )
        .unwrap();
        let cli = super::Cli::try_parse_from([
            "hi",
            "--verify",
            "cargo check",
            "--verify",
            "cargo test",
            "--max-verify-repairs",
            "1",
            "--review",
            "always",
            "--lsp",
            "on",
            "--tool-set",
            "minimal",
        ])
        .unwrap();
        let quality = resolve_quality(&cli, &dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            quality.verification,
            VerificationMode::Explicit(vec![
                hi_agent::VerifyStage::new("verify_1", "cargo check"),
                hi_agent::VerifyStage::new("verify_2", "cargo test"),
            ])
        );
        assert_eq!(quality.max_verify_repairs, 1);
        assert_eq!(quality.review, ReviewPolicy::Always);
        assert_eq!(quality.lsp_mode, LspMode::On);
        assert_eq!(quality.tool_set, ToolSet::Minimal);
        assert_eq!(quality.context_exclusions, vec!["generated/**"]);
    }

    #[test]
    fn removed_quality_flags_are_usage_errors() {
        for flag in ["--auto-verify", "--max-verify", "--minimal-tools"] {
            assert!(
                super::Cli::try_parse_from(["hi", flag]).is_err(),
                "obsolete flag still accepted: {flag}"
            );
        }
    }

    #[test]
    fn empty_verification_commands_are_configuration_errors() {
        let dir = temp_dir_with("");
        let cli = super::Cli::try_parse_from(["hi", "--verify", "   "]).unwrap();
        assert!(
            resolve_quality(&cli, &dir)
                .unwrap_err()
                .to_string()
                .contains("must not be empty")
        );

        std::fs::create_dir_all(dir.join(".hi")).unwrap();
        std::fs::write(
            dir.join(".hi/config.toml"),
            "[quality]\nverification = \"explicit\"\nstages = [\"\"]\n",
        )
        .unwrap();
        let cli = super::Cli::try_parse_from(["hi"]).unwrap();
        assert!(
            resolve_quality(&cli, &dir)
                .unwrap_err()
                .to_string()
                .contains("must not be empty")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cargo_pipeline_runs_compile_gate_before_tests() {
        let dir = temp_dir_with("Cargo.toml");
        let stages = detect_verify_pipeline(&dir);
        // The cheap compile gate must come first so errors localize fast.
        assert_eq!(stages[0].name, "check");
        assert!(stages[0].command.contains("cargo check"));
        assert!(stages.last().unwrap().command.contains("cargo test"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn onboarding_mentions_real_interactive_flags() {
        assert!(
            !super::ONBOARDING.contains("--tui"),
            "there is no --tui flag; the TUI is the default"
        );
        assert!(
            super::ONBOARDING.contains("--plain"),
            "onboarding should point to the actual opt-out flag"
        );
    }

    #[test]
    fn pipenetwork_prefers_provider_specific_api_key_env() {
        assert_eq!(
            ProviderName::Pipenetwork.key_envs(),
            &["PIPENETWORK_API_KEY", "HI_API_KEY", "OPENAI_API_KEY"]
        );
    }

    #[test]
    fn merge_config_keeps_global_default_when_local_omits_one() {
        use super::merge_config;
        let mut global = Config {
            default_profile: Some("default".into()),
            profiles: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "default".into(),
                    Profile {
                        provider: Some(ProviderName::Pipenetwork),
                        model: Some("ipop/coder-balanced".into()),
                        api_key: Some("pipe-key".into()),
                        ..Default::default()
                    },
                );
                m
            },
            ..Default::default()
        };
        let local = Config {
            profiles: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "local".into(),
                    Profile {
                        provider: Some(ProviderName::Ollama),
                        model: Some("qwen2.5-coder".into()),
                        ..Default::default()
                    },
                );
                m
            },
            ..Default::default()
        };

        merge_config(&mut global, local);

        assert_eq!(global.default_profile.as_deref(), Some("default"));
        assert!(global.profiles.contains_key("default"));
        assert!(global.profiles.contains_key("local"));
    }

    #[test]
    fn merge_config_honors_explicit_local_default() {
        use super::merge_config;
        let mut global = Config {
            default_profile: Some("default".into()),
            ..Default::default()
        };
        let local = Config {
            default_profile: Some("local".into()),
            ..Default::default()
        };

        merge_config(&mut global, local);

        assert_eq!(global.default_profile.as_deref(), Some("local"));
    }

    #[test]
    fn curate_skills_defaults_on_for_pipenetwork_only() {
        // Default: on for pipenetwork, off for other providers.
        assert!(curate_skills_default(ProviderName::Pipenetwork, None));
        assert!(!curate_skills_default(ProviderName::Openai, None));
        assert!(!curate_skills_default(ProviderName::Ollama, None));
        // An explicit profile setting always wins, both ways.
        assert!(!curate_skills_default(
            ProviderName::Pipenetwork,
            Some(false)
        ));
        assert!(curate_skills_default(ProviderName::Openai, Some(true)));
    }

    #[test]
    fn explore_subagents_default_on_unless_disabled() {
        // On by default for every provider; an explicit profile setting wins.
        assert!(explore_subagents_default(None));
        assert!(!explore_subagents_default(Some(false)));
        assert!(explore_subagents_default(Some(true)));
    }

    #[test]
    fn planner_model_defaults_to_glm_on_pipenetwork_only() {
        // Default: glm-5.2 on pipenetwork, none elsewhere (the id wouldn't route).
        assert_eq!(
            planner_model_default(ProviderName::Pipenetwork, None).as_deref(),
            Some("pipe/glm-5.2-fast")
        );
        assert_eq!(planner_model_default(ProviderName::Openai, None), None);
        assert_eq!(planner_model_default(ProviderName::Ollama, None), None);
        // An explicit profile value always wins.
        assert_eq!(
            planner_model_default(
                ProviderName::Pipenetwork,
                Some("custom/planner".to_string())
            )
            .as_deref(),
            Some("custom/planner")
        );
        assert_eq!(
            planner_model_default(ProviderName::Openai, Some("x/y".to_string())).as_deref(),
            Some("x/y")
        );
    }

    #[test]
    fn pipenetwork_default_max_tokens_is_bounded_unless_cli_overrides() {
        assert_eq!(
            PIPENETWORK_DEFAULT_MAX_TOKENS, 8192,
            "Pipenetwork coding-agent turns need enough headroom to avoid routine continuation recovery"
        );
        assert_eq!(
            configured_max_tokens(ProviderName::Pipenetwork, None, None),
            PIPENETWORK_DEFAULT_MAX_TOKENS
        );
        assert_eq!(
            configured_max_tokens(ProviderName::Pipenetwork, None, Some(DEFAULT_MAX_TOKENS)),
            PIPENETWORK_DEFAULT_MAX_TOKENS,
            "default-valued profiles should be live-sized at runtime"
        );
        assert_eq!(
            configured_max_tokens(
                ProviderName::Pipenetwork,
                None,
                Some(LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS)
            ),
            PIPENETWORK_DEFAULT_MAX_TOKENS,
            "legacy 2048 profiles must not keep undersizing coding-agent turns"
        );
        assert_eq!(
            configured_max_tokens(ProviderName::Pipenetwork, Some(DEFAULT_MAX_TOKENS), None),
            DEFAULT_MAX_TOKENS,
            "explicit CLI override is honored"
        );
        assert!(
            !max_tokens_is_explicit(ProviderName::Pipenetwork, None, Some(DEFAULT_MAX_TOKENS)),
            "profile default should not block live output sizing"
        );
        assert!(
            !max_tokens_is_explicit(
                ProviderName::Pipenetwork,
                None,
                Some(LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS)
            ),
            "legacy 2048 profile default should not block live output sizing"
        );
        assert!(
            max_tokens_is_explicit(ProviderName::Pipenetwork, Some(2048), None),
            "CLI 2048 is deliberate and should remain explicit"
        );
        assert_eq!(
            configured_max_tokens(ProviderName::Openai, None, None),
            DEFAULT_MAX_TOKENS
        );
    }

    #[test]
    fn pipenetwork_has_default_mcp_url() {
        assert_eq!(
            ProviderName::Pipenetwork.default_mcp_url(),
            Some(hi_ai::PIPE_MCP_DEFAULT_URL)
        );
        assert_eq!(ProviderName::Openai.default_mcp_url(), None);
    }

    #[test]
    fn config_round_trips_through_toml() {
        let mut config = Config {
            default_profile: Some("sonnet".into()),
            ..Default::default()
        };
        config.profiles.insert(
            "sonnet".into(),
            Profile {
                provider: Some(ProviderName::Anthropic),
                model: Some("claude-sonnet-4-20250514".into()),
                mcp_url: Some("https://example.test/mcp".into()),
                api_key_env: Some("ANTHROPIC_API_KEY".into()),
                ..Default::default()
            },
        );
        config.profiles.insert(
            "local".into(),
            Profile {
                provider: Some(ProviderName::Ollama),
                ..Default::default()
            },
        );

        let dir = temp_dir_with("");
        let path = dir.join("config.toml");
        save_config_to(&config, &path).unwrap();

        // Re-read and verify.
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("[profiles.sonnet]"));
        assert!(text.contains("[profiles.local]"));
        assert!(text.contains("provider = \"anthropic\""));
        assert!(text.contains("mcp_url = \"https://example.test/mcp\""));
        assert!(text.contains("api_key_env = \"ANTHROPIC_API_KEY\""));
        // Ollama profile has no model — it should be absent, not `model = ""`.
        // Check just the local section (between [profiles.local] and the next
        // [profiles...] or EOF).
        let local_section = text
            .split("[profiles.local]")
            .nth(1)
            .unwrap_or("")
            .split('[')
            .next()
            .unwrap_or("");
        assert!(
            !local_section.contains("model ="),
            "None fields should be omitted, got: {local_section}"
        );

        let reloaded: Config = toml::from_str(&text).unwrap();
        assert_eq!(reloaded.default_profile.as_deref(), Some("sonnet"));
        assert_eq!(
            reloaded.profiles.get("sonnet").unwrap().provider,
            Some(ProviderName::Anthropic)
        );
        assert_eq!(
            reloaded.profiles.get("sonnet").unwrap().mcp_url.as_deref(),
            Some("https://example.test/mcp")
        );
        assert_eq!(
            reloaded.profiles.get("local").unwrap().provider,
            Some(ProviderName::Ollama)
        );
        assert!(reloaded.profiles.get("local").unwrap().model.is_none());
    }

    #[test]
    fn validate_profile_rejects_endpoint_paths_in_base_url() {
        use super::validate_profile;
        // A bare base URL is fine.
        let ok = Profile {
            provider: Some(ProviderName::Ollama),
            base_url: Some("http://localhost:11434/v1".into()),
            ..Default::default()
        };
        assert!(validate_profile(&ok).is_ok());

        // Trailing slash is tolerated.
        let ok_slash = Profile {
            base_url: Some("http://localhost:11434/v1/".into()),
            ..ok.clone()
        };
        assert!(validate_profile(&ok_slash).is_ok());

        // Common mistake: full endpoint path appended.
        for bad in [
            "http://localhost:11434/v1/chat/completions",
            "http://localhost:11434/v1/completions",
            "https://api.anthropic.com/messages",
        ] {
            let p = Profile {
                base_url: Some(bad.into()),
                ..ok.clone()
            };
            let err = validate_profile(&p).unwrap_err();
            let msg = format!("{err:#}");
            assert!(
                msg.contains("base_url looks like a full endpoint path"),
                "expected rejection for {bad}, got: {msg}"
            );
        }
    }

    #[test]
    fn to_profile_literal_key_is_stored_as_api_key_not_env_ref() {
        // A real API key that happens to be all uppercase + digits + underscores
        // must NOT be mistaken for an env var name. Without an env var by that
        // name set in the environment, to_profile stores it as a literal.
        use super::ProfileForm;
        let form = ProfileForm {
            name: "work".into(),
            provider: ProviderName::Openai,
            api_key: "SK_LIVE_ABC123_XYZ".into(), // looks like an env var name
            store_as_env: true,                   // even if the form said true, to_profile decides
            model: "gpt-4o".into(),
            base_url: String::new(),
        };
        let p = form.to_profile();
        assert_eq!(p.api_key.as_deref(), Some("SK_LIVE_ABC123_XYZ"));
        assert!(
            p.api_key_env.is_none(),
            "literal key must not be stored as env ref"
        );
    }

    #[test]
    fn to_profile_env_var_name_that_is_set_stored_as_env_ref() {
        use super::ProfileForm;
        // Set an env var whose name matches the input.
        let name = "HI_TEST_KEY_FAKE_123";
        // SAFETY: single-threaded test; no other thread reads/writes the env.
        unsafe { std::env::set_var(name, "secret-value") };
        let form = ProfileForm {
            name: "work".into(),
            provider: ProviderName::Openai,
            api_key: name.into(),
            store_as_env: false, // to_profile decides regardless
            model: "gpt-4o".into(),
            base_url: String::new(),
        };
        let p = form.to_profile();
        assert_eq!(p.api_key_env.as_deref(), Some(name));
        assert!(
            p.api_key.is_none(),
            "env var name must not be stored as literal"
        );
        // SAFETY: single-threaded test cleanup.
        unsafe { std::env::remove_var(name) };
    }

    #[test]
    fn to_profile_env_var_name_that_is_not_set_stored_as_literal() {
        // An input that looks like an env var name but no such env var is set
        // is treated as a literal key (the user pasted a key, not a var name).
        use super::ProfileForm;
        let name = "HI_NEVER_SET_KEY_999";
        assert!(
            std::env::var(name).is_err(),
            "precondition: var must not be set"
        );
        let form = ProfileForm {
            name: "work".into(),
            provider: ProviderName::Openai,
            api_key: name.into(),
            store_as_env: true,
            model: "gpt-4o".into(),
            base_url: String::new(),
        };
        let p = form.to_profile();
        assert_eq!(p.api_key.as_deref(), Some(name));
        assert!(p.api_key_env.is_none());
    }

    #[test]
    fn set_profile_model_updates_only_model() {
        use super::{Config, Profile, set_profile_model};
        let dir = std::env::temp_dir().join(format!(
            "hi-set-model-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let mut config = Config {
            default_profile: Some("default".into()),
            profiles: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "default".into(),
                    Profile {
                        provider: Some(ProviderName::Pipenetwork),
                        model: Some("pipe/auto-coder".into()),
                        api_key: Some("test-key".into()),
                        ..Default::default()
                    },
                );
                m
            },
            ..Default::default()
        };

        set_profile_model(&mut config, "default", "ipop/coder-balanced", Some(&path))
            .expect("set model");

        let p = config.profiles.get("default").unwrap();
        assert_eq!(p.model.as_deref(), Some("ipop/coder-balanced"));
        assert_eq!(p.api_key.as_deref(), Some("test-key"));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("model = \"ipop/coder-balanced\""));
        assert!(text.contains("api_key = \"test-key\""));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn layered_test_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hi-layered-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The leak scenario the layered save exists to prevent: a change to a
    /// globally-defined profile must be written to the global file only —
    /// never by dumping the merged view (global API keys included) into the
    /// project-local `hi.toml`.
    #[test]
    fn layered_save_writes_only_the_owning_file() {
        use super::{owning_path_in, read_config_file, rmw_config_file};
        let dir = layered_test_dir("owning");
        let global = dir.join("config.toml");
        let local = dir.join("hi.toml");
        std::fs::write(
            &global,
            "[profiles.work]\nprovider = \"openai\"\nmodel = \"old\"\napi_key = \"sk-secret\"\n\n\
             [profiles.other]\nprovider = \"openai\"\napi_key = \"sk-other\"\n",
        )
        .unwrap();
        std::fs::write(
            &local,
            "[profiles.scratch]\nprovider = \"ollama\"\nmodel = \"m\"\n",
        )
        .unwrap();
        let layers = vec![local.clone(), global.clone()];

        // "work" lives in the global file — that's where the edit must go.
        assert_eq!(owning_path_in(&layers, "work"), Some(global.clone()));
        // "scratch" lives in the local file, which wins the merge.
        assert_eq!(owning_path_in(&layers, "scratch"), Some(local.clone()));

        let local_before = std::fs::read_to_string(&local).unwrap();
        rmw_config_file(&global, |file| {
            file.profiles.get_mut("work").unwrap().model = Some("new-model".into());
        })
        .unwrap();

        // The local file is byte-for-byte untouched — no global profiles or
        // API keys copied into it.
        assert_eq!(std::fs::read_to_string(&local).unwrap(), local_before);
        // The global file has the new model, keeps its own fields, and gained
        // nothing else.
        let global_cfg = read_config_file(&global).unwrap();
        assert_eq!(global_cfg.profiles.len(), 2);
        let work = &global_cfg.profiles["work"];
        assert_eq!(work.model.as_deref(), Some("new-model"));
        assert_eq!(work.api_key.as_deref(), Some("sk-secret"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A profile defined in both layers must be removed from both — deleting
    /// it from one file lets the merge resurrect it from the other on the
    /// next launch.
    #[test]
    fn remove_targets_every_layer_that_defines_the_profile() {
        use super::{layers_defining, read_config_file, rmw_config_file};
        let dir = layered_test_dir("remove");
        let global = dir.join("config.toml");
        let local = dir.join("hi.toml");
        std::fs::write(
            &global,
            "[profiles.dup]\nprovider = \"openai\"\nmodel = \"g\"\n",
        )
        .unwrap();
        std::fs::write(
            &local,
            "[profiles.dup]\nprovider = \"ollama\"\nmodel = \"l\"\n\n\
             [profiles.keep]\nprovider = \"ollama\"\nmodel = \"k\"\n",
        )
        .unwrap();
        let layers = vec![local.clone(), global.clone()];

        let targets = layers_defining(&layers, "dup");
        assert_eq!(targets, vec![local.clone(), global.clone()]);

        // What remove_profile does without an explicit path.
        for path in &targets {
            rmw_config_file(path, |file| {
                file.profiles.remove("dup");
            })
            .unwrap();
        }
        assert!(
            layers_defining(&layers, "dup").is_empty(),
            "no copy left to resurrect"
        );
        let local_cfg = read_config_file(&local).unwrap();
        assert!(
            local_cfg.profiles.contains_key("keep"),
            "unrelated profile kept"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// RMW on a missing file creates it containing only the mutation.
    #[test]
    fn rmw_creates_missing_file_with_only_the_delta() {
        use super::{Profile, read_config_file, rmw_config_file};
        let dir = layered_test_dir("create");
        let path = dir.join("hi.toml");
        rmw_config_file(&path, |file| {
            file.profiles.insert(
                "new".into(),
                Profile {
                    provider: Some(super::ProviderName::Ollama),
                    model: Some("m".into()),
                    ..Default::default()
                },
            );
        })
        .unwrap();
        let cfg = read_config_file(&path).unwrap();
        assert_eq!(cfg.profiles.len(), 1);
        assert!(cfg.default_profile.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_moves_bogus_api_key_env_to_literal() {
        // Simulate a config written by the old buggy wizard: a literal key
        // stored under api_key_env. The migration should move it to api_key.
        use super::{Config, Profile, migrate_api_key_env_to_literal};
        let dir = std::env::temp_dir().join(format!(
            "hi-migrate-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let mut config = Config {
            default_profile: Some("default".into()),
            profiles: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "default".into(),
                    Profile {
                        provider: Some(ProviderName::Pipenetwork),
                        model: Some("ipop/coder-balanced".into()),
                        api_key_env: Some("api_c55ffaeda6574cdb".into()),
                        ..Default::default()
                    },
                );
                m
            },
            ..Default::default()
        };
        // No env var named "api_c55ffaeda6574cdb" is set, so this is bogus.
        assert!(std::env::var("api_c55ffaeda6574cdb").is_err());
        migrate_api_key_env_to_literal(&mut config, &path);
        let p = config.profiles.get("default").unwrap();
        assert_eq!(p.api_key.as_deref(), Some("api_c55ffaeda6574cdb"));
        assert!(p.api_key_env.is_none(), "bogus env ref must be cleared");
        // The config file should have been rewritten with the repair.
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("api_key ="),
            "file should have literal api_key"
        );
        assert!(
            !text.contains("api_key_env"),
            "file should not have api_key_env: {text}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_leaves_legitimate_api_key_env_alone() {
        // A real env var reference (env var is set) must not be migrated.
        use super::{Config, Profile, migrate_api_key_env_to_literal};
        let env_name = "HI_MIGRATE_LEGIT_123";
        unsafe { std::env::set_var(env_name, "real-key-value") };
        let dir = std::env::temp_dir().join(format!(
            "hi-migrate-legit-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let mut config = Config {
            default_profile: Some("default".into()),
            profiles: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "default".into(),
                    Profile {
                        provider: Some(ProviderName::Pipenetwork),
                        api_key_env: Some(env_name.into()),
                        ..Default::default()
                    },
                );
                m
            },
            ..Default::default()
        };
        migrate_api_key_env_to_literal(&mut config, &path);
        let p = config.profiles.get("default").unwrap();
        assert_eq!(p.api_key_env.as_deref(), Some(env_name));
        assert!(
            p.api_key.is_none(),
            "legitimate env ref must not become literal"
        );
        // File should not have been written (no migration needed).
        assert!(
            !path.exists(),
            "file should not be rewritten when no migration"
        );
        unsafe { std::env::remove_var(env_name) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_leaves_unset_env_var_name_in_api_key_env_alone() {
        // An api_key_env that looks like an env var name but the env var isn't
        // set is a legitimate (unfulfilled) reference — don't move it to api_key
        // (that would authenticate with the literal string and get a 401).
        use super::{Config, Profile, migrate_api_key_env_to_literal};
        let env_name = "HI_NEVER_SET_MIGRATE_999";
        assert!(
            std::env::var(env_name).is_err(),
            "precondition: var must not be set"
        );
        let dir = std::env::temp_dir().join(format!(
            "hi-migrate-unset-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let mut config = Config {
            default_profile: Some("default".into()),
            profiles: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "default".into(),
                    Profile {
                        provider: Some(ProviderName::Pipenetwork),
                        api_key_env: Some(env_name.into()),
                        ..Default::default()
                    },
                );
                m
            },
            ..Default::default()
        };
        migrate_api_key_env_to_literal(&mut config, &path);
        let p = config.profiles.get("default").unwrap();
        assert_eq!(
            p.api_key_env.as_deref(),
            Some(env_name),
            "unset env ref must stay"
        );
        assert!(p.api_key.is_none(), "must not become a literal key");
        assert!(!path.exists(), "file should not be rewritten");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_repairs_env_var_name_misplaced_in_api_key() {
        // The previous version of the migration moved an env var name like
        // "HI_API_KEY" from api_key_env to api_key when the env var wasn't set,
        // causing 401s. If the env var IS set, the migration should replace
        // api_key with the env var's value.
        use super::{Config, Profile, migrate_api_key_env_to_literal};
        let env_name = "HI_MIGRATE_REPAIR_123";
        unsafe { std::env::set_var(env_name, "api_realkey_value") };
        let dir = std::env::temp_dir().join(format!(
            "hi-migrate-repair-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let mut config = Config {
            default_profile: Some("default".into()),
            profiles: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "default".into(),
                    Profile {
                        provider: Some(ProviderName::Pipenetwork),
                        api_key: Some(env_name.into()), // env var name in api_key
                        ..Default::default()
                    },
                );
                m
            },
            ..Default::default()
        };
        migrate_api_key_env_to_literal(&mut config, &path);
        let p = config.profiles.get("default").unwrap();
        assert_eq!(
            p.api_key.as_deref(),
            Some("api_realkey_value"),
            "should be replaced with env var value"
        );
        assert!(p.api_key_env.is_none());
        unsafe { std::env::remove_var(env_name) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_moves_unset_env_var_name_from_api_key_back_to_env_ref() {
        // If api_key holds an env var name and the env var is NOT set, move it
        // back to api_key_env so the user gets the right error ("env var … is
        // not set") instead of a 401 from authenticating with the var name.
        use super::{Config, Profile, migrate_api_key_env_to_literal};
        let env_name = "HI_MIGRATE_BACK_999";
        assert!(
            std::env::var(env_name).is_err(),
            "precondition: var must not be set"
        );
        let dir = std::env::temp_dir().join(format!(
            "hi-migrate-back-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let mut config = Config {
            default_profile: Some("default".into()),
            profiles: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "default".into(),
                    Profile {
                        provider: Some(ProviderName::Pipenetwork),
                        api_key: Some(env_name.into()),
                        ..Default::default()
                    },
                );
                m
            },
            ..Default::default()
        };
        migrate_api_key_env_to_literal(&mut config, &path);
        let p = config.profiles.get("default").unwrap();
        assert_eq!(
            p.api_key_env.as_deref(),
            Some(env_name),
            "should move back to env ref"
        );
        assert!(p.api_key.is_none(), "api_key should be cleared");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_drops_standard_env_name_from_buggy_save_config() {
        // The old setup wizard always wrote api_key_env = key_envs().first()
        // (e.g. "HI_API_KEY" for pipenetwork) regardless of what the user pasted.
        // When that env var isn't set, the migration should drop the bogus
        // reference so resolve falls through to the onboarding error, prompting
        // the user to re-enter their key (the new wizard stores it as api_key).
        use super::{Config, Profile, migrate_api_key_env_to_literal};
        let env_name = "HI_API_KEY";
        assert!(
            std::env::var(env_name).is_err(),
            "precondition: HI_API_KEY must not be set"
        );
        let dir = std::env::temp_dir().join(format!(
            "hi-migrate-drop-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let mut config = Config {
            default_profile: Some("default".into()),
            profiles: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "default".into(),
                    Profile {
                        provider: Some(ProviderName::Pipenetwork),
                        model: Some("ipop/coder-balanced".into()),
                        api_key_env: Some(env_name.into()),
                        ..Default::default()
                    },
                );
                m
            },
            ..Default::default()
        };
        migrate_api_key_env_to_literal(&mut config, &path);
        let p = config.profiles.get("default").unwrap();
        assert!(
            p.api_key_env.is_none(),
            "bogus standard env ref must be dropped"
        );
        assert!(p.api_key.is_none(), "no literal key to recover");
        // File should have been rewritten without api_key_env.
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            !text.contains("api_key_env"),
            "file should not have api_key_env: {text}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
