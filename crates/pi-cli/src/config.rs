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
use pi_ai::Registry;
use serde::Deserialize;

/// A minimal agentic coding tool. Works with any OpenAI-compatible endpoint
/// (OpenRouter, terminaili.com, Ollama, llama.cpp, vLLM) or the native
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

    /// Max output tokens per response.
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (e.g. for varying best-of-N candidates).
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Enable reasoning with this thinking-token budget (Anthropic).
    #[arg(long, value_name = "BUDGET")]
    pub thinking: Option<u32>,

    /// Path to a config file (default: ./hi.toml or ~/.config/hi/config.toml).
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Fetch the latest models.dev catalog into the local cache, then exit.
    #[arg(long)]
    pub refresh_models: bool,

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

    /// Use the full-screen terminal UI (experimental).
    #[arg(long)]
    pub tui: bool,

    /// Verification command run after each turn; on failure the model iterates.
    #[arg(long, value_name = "CMD")]
    pub verify: Option<String>,

    /// Auto-detect the project's test command and use it for verification.
    #[arg(long)]
    pub auto_verify: bool,

    /// Max verification retry rounds.
    #[arg(long, default_value_t = 3)]
    pub max_verify: u32,

    /// Safety cap on model calls per turn (stops runaway tool loops).
    #[arg(long, default_value_t = 50)]
    pub max_steps: u32,

    /// Run N candidate attempts in isolated git worktrees and keep the first
    /// that passes verification. Requires --verify/--auto-verify and a prompt.
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub best_of: u32,

    /// Write a JSON usage/outcome report to this path (for eval/automation).
    #[arg(long, value_name = "PATH")]
    pub report: Option<PathBuf>,

    /// One-shot prompt. If omitted, starts an interactive session.
    pub prompt: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ProviderName {
    /// OpenAI-compatible Chat Completions.
    Openai,
    /// Native Anthropic Messages API.
    Anthropic,
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
    /// Name of the env var holding the API key for this profile.
    pub api_key_env: Option<String>,
    pub max_tokens: Option<u32>,
    pub thinking_budget: Option<u32>,
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

    let provider = cli
        .provider
        .or(profile.and_then(|p| p.provider))
        .unwrap_or(ProviderName::Openai);

    let model = cli
        .model
        .clone()
        .or_else(|| profile.and_then(|p| p.model.clone()))
        .or_else(|| std::env::var("HI_MODEL").ok())
        .ok_or_else(|| anyhow!("no model: pass --model, set HI_MODEL, or define a profile"))?;

    let base_url = cli
        .base_url
        .clone()
        .or_else(|| profile.and_then(|p| p.base_url.clone()))
        .or_else(|| std::env::var("HI_BASE_URL").ok())
        .unwrap_or_else(|| default_base_url(provider).to_string());

    let api_key = resolve_api_key(cli, profile, provider)?;

    let mut max_tokens = cli
        .max_tokens
        .or(profile.and_then(|p| p.max_tokens))
        .unwrap_or(4096);
    // Don't exceed a known model's output ceiling (avoids a 400 from Anthropic).
    if let Some(info) = registry.lookup(&model)
        && info.max_output > 0
        && max_tokens > info.max_output
    {
        max_tokens = info.max_output;
    }

    let thinking_budget = cli.thinking.or(profile.and_then(|p| p.thinking_budget));

    Ok(Settings {
        provider,
        model,
        base_url,
        api_key,
        max_tokens,
        thinking_budget,
    })
}

/// Guess the project's test command from marker files in `dir`. Used by
/// `--auto-verify` so the proven verify-loop is zero-config.
pub fn detect_verify_command_in(dir: &Path) -> Option<String> {
    let has = |name: &str| dir.join(name).exists();
    if has("Cargo.toml") {
        Some("cargo test".into())
    } else if has("go.mod") {
        Some("go test ./...".into())
    } else if has("pyproject.toml") || has("setup.py") || has("pytest.ini") || has("tox.ini") {
        Some("pytest -q".into())
    } else if has("package.json") {
        Some("npm test".into())
    } else if has("Makefile") || has("makefile") {
        Some("make test".into())
    } else {
        None
    }
}

fn default_base_url(provider: ProviderName) -> &'static str {
    match provider {
        ProviderName::Openai => "https://openrouter.ai/api/v1",
        ProviderName::Anthropic => "https://api.anthropic.com",
    }
}

fn resolve_api_key(cli: &Cli, profile: Option<&Profile>, provider: ProviderName) -> Result<String> {
    if let Some(key) = &cli.api_key {
        return Ok(key.clone());
    }
    if let Some(env_name) = profile.and_then(|p| p.api_key_env.as_ref()) {
        return std::env::var(env_name)
            .map_err(|_| anyhow!("env var {env_name} (from profile) is not set"));
    }
    let candidates: &[&str] = match provider {
        ProviderName::Anthropic => &["HI_API_KEY", "ANTHROPIC_API_KEY"],
        ProviderName::Openai => &["HI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"],
    };
    for name in candidates {
        if let Ok(value) = std::env::var(name)
            && !value.is_empty()
        {
            return Ok(value);
        }
    }
    bail!("no API key: pass --api-key or set one of {candidates:?}");
}

#[cfg(test)]
mod tests {
    use super::detect_verify_command_in;
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
    fn detects_test_commands_by_marker() {
        let cases = [
            ("Cargo.toml", Some("cargo test")),
            ("go.mod", Some("go test ./...")),
            ("pyproject.toml", Some("pytest -q")),
            ("package.json", Some("npm test")),
            ("Makefile", Some("make test")),
            ("", None),
        ];
        for (marker, expected) in cases {
            let dir = temp_dir_with(marker);
            let got = detect_verify_command_in(&dir);
            assert_eq!(got.as_deref(), expected, "marker={marker:?}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
