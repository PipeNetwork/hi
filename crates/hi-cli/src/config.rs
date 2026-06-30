//! CLI parsing, config-file profiles, and resolution into effective settings.
//!
//! Precedence, highest first: explicit CLI flags → selected profile → env vars
//! → built-in defaults. Profiles let a user keep several models on hand
//! (e.g. a cloud Anthropic profile and a local Ollama profile) and switch with
//! `-p <name>`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use hi_agent::VerifyStage;
use hi_ai::{CompatMode, Registry, ToolMode};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_TOKENS: u32 = 8192;
const PIPENETWORK_DEFAULT_MAX_TOKENS: u32 = 2048;

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

    /// Tool calling mode: auto, required, chat-only, or read-only.
    #[arg(long, value_enum)]
    pub tool_mode: Option<CliToolMode>,

    /// Provider compatibility policy: auto retries simpler request shapes; strict sends one shape.
    #[arg(long, value_enum)]
    pub compat: Option<CliCompatMode>,

    /// Path to a config file (default: ./hi.toml or ~/.config/hi/config.toml).
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Fetch the latest models.dev catalog into the local cache, then exit.
    #[arg(long)]
    pub refresh_models: bool,

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

    /// List saved sessions, then exit.
    #[arg(long)]
    pub list_sessions: bool,

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

    /// Verification command run after each turn; on failure the model iterates.
    #[arg(long, value_name = "CMD")]
    pub verify: Option<String>,

    /// Auto-detect the project's test command and use it for verification.
    #[arg(long)]
    pub auto_verify: bool,

    /// Max verification retry rounds.
    #[arg(long, default_value_t = 2)]
    pub max_verify: u32,

    /// Safety cap on model calls per turn (stops runaway tool loops).
    #[arg(long, default_value_t = 500)]
    pub max_steps: u32,

    /// Run N candidate attempts in isolated git worktrees and keep the first
    /// that passes verification. Requires --verify/--auto-verify and a prompt.
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub best_of: u32,

    /// Write a JSON usage/outcome report to this path (for eval/automation).
    #[arg(long, value_name = "PATH")]
    pub report: Option<PathBuf>,

    /// Warn when cumulative session cost exceeds this USD amount (e.g. 0.50).
    /// The warning is shown after each turn; the turn is not interrupted.
    #[arg(long, value_name = "USD")]
    pub max_cost: Option<f64>,

    /// Quiet: print only the assistant's text (no tool chatter or usage line).
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// One-shot prompt. If omitted, starts an interactive session.
    pub prompt: Option<String>,
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
            ProviderName::Pipenetwork => &["HI_API_KEY", "PIPENETWORK_API_KEY", "OPENAI_API_KEY"],
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
    pub profiles: HashMap<String, Profile>,
}

impl serde::Serialize for Config {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Config", 2)?;
        if let Some(v) = &self.default_profile {
            s.serialize_field("default_profile", v)?;
        }
        if !self.profiles.is_empty() {
            // BTreeMap serializes as a sorted map → stable, alphabetical output.
            let sorted: BTreeMap<&String, &Profile> = self.profiles.iter().collect();
            s.serialize_field("profiles", &sorted)?;
        }
        s.end()
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Profile {
    pub provider: Option<ProviderName>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    /// MCP endpoint used for metadata discovery, when supported by the provider.
    pub mcp_url: Option<String>,
    /// A literal API key (written by the setup wizard).
    pub api_key: Option<String>,
    /// Name of an env var holding the API key for this profile.
    pub api_key_env: Option<String>,
    pub max_tokens: Option<u32>,
    pub thinking_budget: Option<u32>,
    #[serde(default)]
    pub tool_mode: Option<ToolMode>,
    #[serde(default)]
    pub compat: Option<CompatMode>,
    /// Other profile names to fall back to, in order, when this one returns
    /// nothing or errors.
    pub fallback: Option<Vec<String>>,
}

// Serialize `Profile` with clean output: omit `None` fields so the TOML
// doesn't fill with `model = ""` lines. We can't put `skip_serializing_if` on
// each field above (it'd require repeating it 9 times), so we implement a
// custom serializer that skips None values.
impl serde::Serialize for Profile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Profile", 11)?;
        if let Some(v) = &self.provider {
            s.serialize_field("provider", v)?;
        }
        if let Some(v) = &self.model {
            s.serialize_field("model", v)?;
        }
        if let Some(v) = &self.base_url {
            s.serialize_field("base_url", v)?;
        }
        if let Some(v) = &self.mcp_url {
            s.serialize_field("mcp_url", v)?;
        }
        if let Some(v) = &self.api_key {
            s.serialize_field("api_key", v)?;
        }
        if let Some(v) = &self.api_key_env {
            s.serialize_field("api_key_env", v)?;
        }
        if let Some(v) = &self.max_tokens {
            s.serialize_field("max_tokens", v)?;
        }
        if let Some(v) = &self.thinking_budget {
            s.serialize_field("thinking_budget", v)?;
        }
        if let Some(v) = &self.tool_mode {
            s.serialize_field("tool_mode", v)?;
        }
        if let Some(v) = &self.compat {
            s.serialize_field("compat", v)?;
        }
        if let Some(v) = &self.fallback {
            s.serialize_field("fallback", v)?;
        }
        s.end()
    }
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
    pub thinking_budget: Option<u32>,
    pub tool_mode: ToolMode,
    pub compat: CompatMode,
}

pub fn load_config(explicit: Option<&Path>) -> Result<Config> {
    let Some(path) = config_path(explicit) else {
        return Ok(Config::default());
    };
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading config {}", path.display()))?;
    let mut config = toml::from_str::<Config>(&text)
        .with_context(|| format!("parsing config {}", path.display()))?;
    migrate_api_key_env_to_literal(&mut config, &path);
    Ok(config)
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
                // The env var is set — use its value as the literal key.
                profile.api_key = Some(val);
                changed = true;
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
    if changed {
        // Best-effort rewrite; if it fails we've still repaired the in-memory
        // config so this run works, just not the next one.
        if let Ok(text) = toml::to_string_pretty(config) {
            let _ = std::fs::write(path, text);
        }
    }
}

fn config_path(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    let local = PathBuf::from("hi.toml");
    if local.exists() {
        return Some(local);
    }
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    let candidate = base.join("hi").join("config.toml");
    candidate.exists().then_some(candidate)
}

/// Apply precedence to produce the effective [`Settings`].
pub fn resolve(cli: &Cli, config: &Config, registry: &Registry) -> Result<Settings> {
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

    let mut max_tokens =
        configured_max_tokens(provider, cli.max_tokens, profile.and_then(|p| p.max_tokens));
    // Don't exceed a known model's output ceiling (avoids a 400 from Anthropic).
    if let Some(info) = registry.lookup(&model)
        && info.max_output > 0
        && max_tokens > info.max_output
    {
        max_tokens = info.max_output;
    }

    let thinking_budget = cli.thinking.or(profile.and_then(|p| p.thinking_budget));
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

    Ok(Settings {
        provider,
        model,
        base_url,
        mcp_url,
        api_key,
        max_tokens,
        thinking_budget,
        tool_mode,
        compat,
    })
}

/// Bounds on the session-start repo map, to keep it useful without flooding the
/// context window every turn. Kept tight since the system prompt is resent
/// on every model call — each line costs tokens × rounds.
const MAP_MAX_FILES: usize = 25;
const MAP_MAX_PER_FILE: usize = 8;
const MAP_MAX_LINES: usize = 80;

/// A heuristic "repo map" for the system prompt: each source file followed by
/// its top-level declarations (functions, types, classes…), so the model can
/// navigate without reading everything first. Signature-only and bounded; this
/// is a cheap stand-in for a tree-sitter map, and returns `None` outside a git
/// repo or when nothing is found.
pub fn build_repo_map(dir: &Path) -> Option<String> {
    let files = git_source_files(dir)?;
    let mut body = String::new();
    let mut lines_used = 0;
    let mut files_shown = 0;
    for file in files.iter().take(MAP_MAX_FILES) {
        if lines_used >= MAP_MAX_LINES {
            break;
        }
        let Ok(content) = std::fs::read_to_string(dir.join(file)) else {
            continue;
        };
        let sigs: Vec<String> = content
            .lines()
            .filter(|l| looks_like_signature(l))
            .take(MAP_MAX_PER_FILE)
            .map(|l| clip_line(l.trim()))
            .collect();
        if sigs.is_empty() {
            continue;
        }
        body.push_str(file);
        body.push('\n');
        for s in sigs {
            if lines_used >= MAP_MAX_LINES {
                break;
            }
            body.push_str("  ");
            body.push_str(&s);
            body.push('\n');
            lines_used += 1;
        }
        files_shown += 1;
    }
    (files_shown > 0).then(|| format!("# Repo map (heuristic — top declarations per file)\n{body}"))
}

/// Git-tracked source files (by extension), sorted. `None` outside a git repo.
fn git_source_files(dir: &Path) -> Option<Vec<String>> {
    const EXTS: &[&str] = &[
        "rs", "go", "py", "js", "jsx", "ts", "tsx", "java", "kt", "c", "cc", "cpp", "h", "hpp",
        "rb", "swift", "scala", "cs", "php",
    ];
    let out = std::process::Command::new("git")
        .arg("ls-files")
        .current_dir(dir)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let mut files: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|f| {
            Path::new(f)
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| EXTS.contains(&e))
        })
        .map(str::to_string)
        .collect();
    files.sort();
    (!files.is_empty()).then_some(files)
}

/// Whether a line declares something worth mapping (a fn/type/class/etc.),
/// after stripping leading visibility/async modifiers.
fn looks_like_signature(line: &str) -> bool {
    let mut t = line.trim();
    for kw in [
        "pub",
        "export",
        "default",
        "async",
        "static",
        "final",
        "public",
        "private",
        "protected",
        "unsafe",
        "abstract",
    ] {
        if let Some(rest) = t.strip_prefix(kw)
            && rest.starts_with(char::is_whitespace)
        {
            t = rest.trim_start();
        }
    }
    const DECL: &[&str] = &[
        "fn ",
        "func ",
        "def ",
        "struct ",
        "enum ",
        "trait ",
        "impl ",
        "impl<",
        "class ",
        "interface ",
        "type ",
        "mod ",
        "module ",
        "function ",
    ];
    DECL.iter().any(|d| t.starts_with(d))
}

/// Trim a signature line for the map: drop a trailing `{` and clip length.
fn clip_line(s: &str) -> String {
    let s = s.trim_end().trim_end_matches('{').trim_end();
    if s.chars().count() > 100 {
        format!("{}…", s.chars().take(100).collect::<String>())
    } else {
        s.to_string()
    }
}

/// Guess a *layered* verification pipeline from marker files in `dir`: a cheap
/// compile/typecheck (and lint, when obviously configured) before tests, so the
/// model gets fast, localizable errors before the slower test stage. Used by
/// `--auto-verify` so the proven verify-loop is zero-config. Empty = unknown
/// project.
pub fn detect_verify_pipeline(dir: &Path) -> Vec<VerifyStage> {
    let has = |name: &str| dir.join(name).exists();
    let stage = |name: &str, cmd: &str| VerifyStage::new(name, cmd);
    if has("Cargo.toml") {
        // `cargo check` fails faster and reports cleaner compiler errors than
        // compiling the test harness; `cargo test` then covers behavior.
        vec![
            stage("check", "cargo check --quiet"),
            stage("test", "cargo test --quiet"),
        ]
    } else if has("go.mod") {
        vec![
            stage("build", "go build ./..."),
            stage("test", "go test ./..."),
        ]
    } else if has("package.json") {
        // Type-check first when a tsconfig is present (catches what jest won't).
        let mut v = Vec::new();
        if has("tsconfig.json") {
            v.push(stage("typecheck", "npx --no-install tsc --noEmit"));
        }
        v.push(stage("test", "npm test --silent"));
        v
    } else if has("pyproject.toml") || has("setup.py") || has("pytest.ini") || has("tox.ini") {
        // Add a ruff lint gate only when ruff is clearly configured, to avoid
        // false failures on projects that don't use it.
        let mut v = Vec::new();
        if has("ruff.toml") || has(".ruff.toml") {
            v.push(stage("lint", "ruff check ."));
        }
        v.push(stage("test", "pytest -q"));
        v
    } else if has("Makefile") || has("makefile") {
        vec![stage("test", "make test")]
    } else {
        Vec::new()
    }
}

/// True when nothing is configured — used to trigger the interactive setup
/// wizard on a fresh terminal.
pub fn needs_setup(cli: &Cli, file: &Config) -> bool {
    cli.model.is_none()
        && cli.provider.is_none()
        && cli.profile.is_none()
        && file.default_profile.is_none()
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

/// Add or replace a profile in the config and save it to disk.
pub fn upsert_profile(
    config: &mut Config,
    name: &str,
    profile: Profile,
    config_path: &Path,
) -> Result<()> {
    validate_profile(&profile)?;
    config.profiles.insert(name.to_string(), profile);
    save_config_to(config, config_path)
}

/// Remove a profile from the config and save it to disk. Returns `false` if the
/// profile didn't exist (caller may treat that as an error or a no-op).
pub fn remove_profile(config: &mut Config, name: &str, config_path: &Path) -> Result<bool> {
    if config.profiles.remove(name).is_none() {
        return Ok(false);
    }
    save_config_to(config, config_path)?;
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
pub fn resolve_fallbacks(cli: &Cli, config: &Config, registry: &Registry) -> Vec<Settings> {
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
        match resolve_named_profile(config, &name, registry) {
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
/// profiles at startup and for `/provider` switches mid-session.
///
/// If the profile has no `model` and the provider has no default, a placeholder
/// is used — the caller is expected to pick a real model via `/model` from what
/// the endpoint serves. The placeholder is fine for building the provider and
/// listing models, but a turn can't run with it.
pub fn resolve_named_profile(config: &Config, name: &str, registry: &Registry) -> Result<Settings> {
    let profile = config
        .profiles
        .get(name)
        .ok_or_else(|| anyhow!("profile '{name}' not found in config"))?;

    let provider = profile.provider.unwrap_or(ProviderName::Openai);
    let model = profile
        .model
        .clone()
        .or_else(|| provider.default_model().map(String::from))
        .unwrap_or_else(|| "__pick_via_model__".to_string());
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

    let mut max_tokens = configured_max_tokens(provider, None, profile.max_tokens);
    if let Some(info) = registry.lookup(&model)
        && info.max_output > 0
        && max_tokens > info.max_output
    {
        max_tokens = info.max_output;
    }

    Ok(Settings {
        provider,
        model,
        base_url,
        mcp_url,
        api_key,
        max_tokens,
        thinking_budget: profile.thinking_budget,
        tool_mode: profile.tool_mode.unwrap_or_default(),
        compat: profile.compat.unwrap_or_default(),
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
        // The setup wizard historically wrote `8192` into pipenetwork profiles.
        // Treat that legacy value as the old default and use the bounded hosted
        // route budget instead; an explicit CLI --max-tokens still wins above.
        (ProviderName::Pipenetwork, None | Some(DEFAULT_MAX_TOKENS)) => {
            PIPENETWORK_DEFAULT_MAX_TOKENS
        }
        (_, Some(value)) => value,
        (_, None) => DEFAULT_MAX_TOKENS,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Config, DEFAULT_MAX_TOKENS, PIPENETWORK_DEFAULT_MAX_TOKENS, Profile, ProviderName,
        configured_max_tokens, detect_verify_pipeline, save_config_to,
    };
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
    fn signature_detection_skips_non_decls() {
        use super::looks_like_signature;
        assert!(looks_like_signature("pub fn run() {"));
        assert!(looks_like_signature("    async fn helper(x: u8) -> u8 {"));
        assert!(looks_like_signature("struct App {"));
        assert!(looks_like_signature("def parse(s):"));
        assert!(looks_like_signature("export function main() {"));
        // Not declarations.
        assert!(!looks_like_signature("    let x = 1;"));
        assert!(!looks_like_signature("// a comment"));
        assert!(!looks_like_signature("return fn_result;"));
    }

    #[test]
    fn repo_map_lists_signatures_for_a_git_repo() {
        use super::build_repo_map;
        // The hi repo itself is a git repo with Rust sources.
        let map = build_repo_map(std::path::Path::new(".."))
            .or_else(|| build_repo_map(std::path::Path::new(".")));
        if let Some(map) = map {
            assert!(
                map.contains("Repo map"),
                "has a header: {}",
                &map[..map.len().min(80)]
            );
            assert!(map.contains("fn "), "lists function signatures");
        }
        // (No panic / sane output is the assertion; outside git it returns None.)
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
    fn pipenetwork_default_max_tokens_is_bounded_unless_cli_overrides() {
        assert_eq!(
            configured_max_tokens(ProviderName::Pipenetwork, None, None),
            PIPENETWORK_DEFAULT_MAX_TOKENS
        );
        assert_eq!(
            configured_max_tokens(ProviderName::Pipenetwork, None, Some(DEFAULT_MAX_TOKENS)),
            PIPENETWORK_DEFAULT_MAX_TOKENS,
            "legacy wizard profiles should not keep using 8192 by default"
        );
        assert_eq!(
            configured_max_tokens(ProviderName::Pipenetwork, Some(DEFAULT_MAX_TOKENS), None),
            DEFAULT_MAX_TOKENS,
            "explicit CLI override is honored"
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
