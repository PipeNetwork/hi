//! CLI parsing, config-file profiles, and resolution into effective settings.
//!
//! Precedence, highest first: explicit CLI flags → selected profile → env vars
//! → built-in defaults. Profiles let a user keep several models on hand
//! (e.g. a cloud Anthropic profile and a local Ollama profile) and switch with
//! `-p <name>`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use hi_agent::VerifyStage;
use hi_ai::{CompatMode, Registry, ToolMode};
use serde::Deserialize;

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

    /// Quiet: print only the assistant's text (no tool chatter or usage line).
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// One-shot prompt. If omitted, starts an interactive session.
    pub prompt: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, ValueEnum)]
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

#[derive(Debug, Default, Deserialize)]
pub struct Config {
    pub default_profile: Option<String>,
    #[serde(default)]
    pub profiles: HashMap<String, Profile>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Profile {
    pub provider: Option<ProviderName>,
    pub model: Option<String>,
    pub base_url: Option<String>,
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

/// Fully-resolved settings used to build a provider and run the agent.
#[derive(Debug)]
pub struct Settings {
    pub provider: ProviderName,
    pub model: String,
    pub base_url: String,
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
    toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
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

    let api_key = resolve_api_key(cli, profile, provider)?;

    let mut max_tokens = cli
        .max_tokens
        .or(profile.and_then(|p| p.max_tokens))
        .unwrap_or(8192);
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

/// Shown when `hi` is run with nothing configured. Actionable, not terse.
const ONBOARDING: &str = "no model configured. Get started with one of:

  Local (Ollama):   hi --provider ollama -m qwen2.5-coder \"...\"
  Anthropic:        ANTHROPIC_API_KEY=...   hi --provider anthropic \"...\"
  OpenRouter:       OPENROUTER_API_KEY=...  hi -m anthropic/claude-sonnet-4 \"...\"
  pipenetwork.ai:   PIPENETWORK_API_KEY=...  hi --provider pipenetwork \"...\"

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
        return std::env::var(env_name)
            .map_err(|_| anyhow!("env var {env_name} (from profile) is not set"));
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
/// (no CLI overrides — those belong to the primary).
fn resolve_named_profile(config: &Config, name: &str, registry: &Registry) -> Result<Settings> {
    let profile = config
        .profiles
        .get(name)
        .ok_or_else(|| anyhow!("profile '{name}' not found in config"))?;

    let provider = profile.provider.unwrap_or(ProviderName::Openai);
    let model = profile
        .model
        .clone()
        .or_else(|| provider.default_model().map(String::from))
        .ok_or_else(|| anyhow!("no model set"))?;
    let base_url = profile
        .base_url
        .clone()
        .unwrap_or_else(|| provider.default_base_url().to_string());
    let api_key = resolve_api_key_for(Some(profile), provider)?;

    let mut max_tokens = profile.max_tokens.unwrap_or(8192);
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
        api_key,
        max_tokens,
        thinking_budget: profile.thinking_budget,
        tool_mode: profile.tool_mode.unwrap_or_default(),
        compat: profile.compat.unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::detect_verify_pipeline;
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
}
