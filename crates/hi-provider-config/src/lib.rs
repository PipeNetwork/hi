//! Shared, provider-only projection of `hi` configuration.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    str::FromStr,
};

use serde::Deserialize;
use thiserror::Error;

pub const DEFAULT_MAX_TOKENS: u32 = 8192;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderName {
    Openai,
    Anthropic,
    Pipenetwork,
    Ollama,
    Xai,
}

impl ProviderName {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Pipenetwork => "pipenetwork",
            Self::Ollama => "ollama",
            Self::Xai => "xai",
        }
    }

    pub const fn default_base_url(self) -> &'static str {
        match self {
            Self::Openai => "https://openrouter.ai/api/v1",
            Self::Anthropic => "https://api.anthropic.com",
            Self::Pipenetwork => "https://api.pipenetwork.ai/v1",
            Self::Ollama => "http://localhost:11434/v1",
            Self::Xai => "https://api.x.ai/v1",
        }
    }

    pub const fn default_model(self) -> Option<&'static str> {
        match self {
            Self::Pipenetwork => Some("ipop/coder-balanced"),
            Self::Anthropic => Some("claude-opus-4-8"),
            Self::Xai => Some("grok-4.3"),
            _ => None,
        }
    }

    pub const fn default_mcp_url(self) -> Option<&'static str> {
        match self {
            // Must stay equal to `hi_ai::PIPE_MCP_DEFAULT_URL` (this crate sits
            // below hi-ai, so it cannot reference the constant); hi-shell has a
            // test tripwire asserting the two agree.
            Self::Pipenetwork => Some("https://api.pipenetwork.ai/mcp"),
            _ => None,
        }
    }

    pub const fn key_envs(self) -> &'static [&'static str] {
        match self {
            Self::Anthropic => &["HI_API_KEY", "ANTHROPIC_API_KEY"],
            Self::Pipenetwork => &["PIPENETWORK_API_KEY", "HI_API_KEY", "OPENAI_API_KEY"],
            Self::Ollama => &["HI_API_KEY", "OLLAMA_API_KEY"],
            Self::Openai => &["HI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"],
            Self::Xai => &["XAI_API_KEY", "HI_API_KEY"],
        }
    }
}

impl FromStr for ProviderName {
    type Err = ConfigError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "openai" => Ok(Self::Openai),
            "anthropic" => Ok(Self::Anthropic),
            "pipenetwork" | "pipe" => Ok(Self::Pipenetwork),
            "ollama" | "local" => Ok(Self::Ollama),
            "xai" => Ok(Self::Xai),
            other => Err(ConfigError::UnknownProvider(other.to_owned())),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub struct ProviderProfile {
    pub provider: Option<ProviderName>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub max_tokens: Option<u32>,
    pub mcp_url: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub struct ProviderConfigFile {
    pub default_profile: Option<String>,
    #[serde(default)]
    pub profiles: HashMap<String, ProviderProfile>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResolveOverrides {
    pub profile: Option<String>,
    pub provider: Option<ProviderName>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub max_tokens: Option<u32>,
    pub mcp_url: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedProviderConfig {
    pub profile: Option<String>,
    pub provider: ProviderName,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub max_tokens: u32,
    pub mcp_url: Option<String>,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("reading provider config {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("parsing provider config {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("profile '{0}' not found in config")]
    MissingProfile(String),
    #[error("unknown provider '{0}'")]
    UnknownProvider(String),
    #[error("no model configured; set HI_MODEL or select a profile with a model")]
    MissingModel,
    #[error("no API key configured; set HI_API_KEY or a provider API key variable")]
    MissingApiKey,
}

pub fn global_config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join("hi").join("config.toml"))
}

/// Load provider config with the same semantics as `hi` itself: the global
/// file first, then a project-local `hi.toml` merged on top. A local file that
/// only sets, say, `[sync]` must not erase the user's global profiles and
/// default provider.
pub fn load(path: Option<&Path>) -> Result<ProviderConfigFile, ConfigError> {
    if let Some(path) = path {
        return load_file(path);
    }
    let mut config = match global_config_path() {
        Some(path) if path.exists() => load_file(&path)?,
        _ => ProviderConfigFile::default(),
    };
    let local = PathBuf::from("hi.toml");
    if local.exists() {
        let overlay = load_file(&local)?;
        if overlay.default_profile.is_some() {
            config.default_profile = overlay.default_profile;
        }
        config.profiles.extend(overlay.profiles);
    }
    Ok(config)
}

fn load_file(path: &Path) -> Result<ProviderConfigFile, ConfigError> {
    if !path.exists() {
        return Ok(ProviderConfigFile::default());
    }
    let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    toml::from_str(&text).map_err(|source| ConfigError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

pub fn resolve(
    file: &ProviderConfigFile,
    explicit: ResolveOverrides,
) -> Result<ResolvedProviderConfig, ConfigError> {
    resolve_with_env(file, explicit, |name| std::env::var(name).ok())
}

fn resolve_with_env(
    file: &ProviderConfigFile,
    explicit: ResolveOverrides,
    env: impl Fn(&str) -> Option<String>,
) -> Result<ResolvedProviderConfig, ConfigError> {
    let profile_name = explicit
        .profile
        .clone()
        .or_else(|| file.default_profile.clone());
    let profile = profile_name
        .as_ref()
        .map(|name| {
            file.profiles
                .get(name)
                .ok_or_else(|| ConfigError::MissingProfile(name.clone()))
        })
        .transpose()?;

    let provider = match explicit
        .provider
        .or_else(|| profile.and_then(|value| value.provider))
    {
        Some(provider) => provider,
        // A typo in HI_PROVIDER must fail loudly: silently falling back to the
        // default provider would send whatever API key resolves to a host the
        // user never chose.
        None => match env("HI_PROVIDER") {
            Some(value) => value.parse()?,
            None => ProviderName::Openai,
        },
    };
    let model = explicit
        .model
        .or_else(|| profile.and_then(|value| value.model.clone()))
        .or_else(|| env("HI_MODEL"))
        .or_else(|| provider.default_model().map(str::to_owned))
        .ok_or(ConfigError::MissingModel)?;
    let base_url = explicit
        .base_url
        .or_else(|| profile.and_then(|value| value.base_url.clone()))
        .or_else(|| env("HI_BASE_URL"))
        .unwrap_or_else(|| provider.default_base_url().to_owned());
    let mcp_url = explicit
        .mcp_url
        .or_else(|| profile.and_then(|value| value.mcp_url.clone()))
        .or_else(|| env("HI_MCP_URL"))
        .or_else(|| provider.default_mcp_url().map(str::to_owned))
        .filter(|value| !value.trim().is_empty());
    let api_key = explicit
        .api_key
        .or_else(|| profile.and_then(|value| value.api_key.clone()))
        .or_else(|| {
            let reference = profile.and_then(|value| value.api_key_env.as_deref())?;
            if let Some(value) = env(reference) {
                return Some(value);
            }
            // Tolerate wizard-damaged configs that stored a pasted literal key
            // in `api_key_env` (hi repairs these on load; this loader must not
            // silently authenticate with the literal string of an env name
            // either): a value that cannot be an env-var name is the key.
            let looks_like_env_name = !reference.is_empty()
                && reference
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
            (!looks_like_env_name).then(|| reference.to_string())
        })
        .or_else(|| provider.key_envs().iter().find_map(|name| env(name)))
        // Ollama needs no key; xAI may authenticate through the signed-in token
        // store, which the provider builder selects when the key is empty (and
        // reports "sign in" guidance when the store is empty too). Erroring
        // here would make that path unreachable.
        .or_else(|| {
            matches!(provider, ProviderName::Ollama | ProviderName::Xai).then(String::new)
        })
        .ok_or(ConfigError::MissingApiKey)?;
    let max_tokens = explicit
        .max_tokens
        .or_else(|| profile.and_then(|value| value.max_tokens))
        .or_else(|| env("HI_MAX_TOKENS").and_then(|value| value.parse().ok()))
        .unwrap_or(DEFAULT_MAX_TOKENS);

    Ok(ResolvedProviderConfig {
        profile: profile_name,
        provider,
        model,
        base_url,
        api_key,
        max_tokens,
        mcp_url,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(values: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[test]
    fn precedence_is_explicit_profile_environment_defaults() {
        let file: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "work"
[profiles.work]
provider = "anthropic"
model = "profile-model"
base_url = "https://profile"
api_key = "profile-key"
max_tokens = 100
mcp_url = "https://profile-mcp"
"#,
        )
        .unwrap();
        let explicit = ResolveOverrides {
            model: Some("explicit-model".into()),
            max_tokens: Some(300),
            ..Default::default()
        };
        let resolved = resolve_with_env(
            &file,
            explicit,
            env(&[
                ("HI_MODEL", "env-model"),
                ("HI_BASE_URL", "https://env"),
                ("HI_API_KEY", "env-key"),
                ("HI_MAX_TOKENS", "200"),
            ]),
        )
        .unwrap();
        assert_eq!(resolved.provider, ProviderName::Anthropic);
        assert_eq!(resolved.model, "explicit-model");
        assert_eq!(resolved.base_url, "https://profile");
        assert_eq!(resolved.api_key, "profile-key");
        assert_eq!(resolved.max_tokens, 300);
        assert_eq!(resolved.mcp_url.as_deref(), Some("https://profile-mcp"));
    }

    #[test]
    fn explicit_profile_beats_default_profile() {
        let file: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "one"
[profiles.one]
provider = "anthropic"
model = "one"
api_key = "one-key"
[profiles.two]
provider = "xai"
model = "two"
api_key = "two-key"
"#,
        )
        .unwrap();
        let resolved = resolve_with_env(
            &file,
            ResolveOverrides {
                profile: Some("two".into()),
                ..Default::default()
            },
            env(&[]),
        )
        .unwrap();
        assert_eq!(resolved.profile.as_deref(), Some("two"));
        assert_eq!(resolved.provider, ProviderName::Xai);
        assert_eq!(resolved.model, "two");
    }

    #[test]
    fn unknown_hi_provider_errors_instead_of_defaulting() {
        let result = resolve_with_env(
            &ProviderConfigFile::default(),
            ResolveOverrides::default(),
            env(&[
                ("HI_PROVIDER", "Anthropic"),
                ("HI_API_KEY", "key"),
                ("HI_MODEL", "m"),
            ]),
        );
        assert!(matches!(result, Err(ConfigError::UnknownProvider(_))));
    }

    #[test]
    fn xai_without_key_resolves_empty_for_token_store() {
        let resolved = resolve_with_env(
            &ProviderConfigFile::default(),
            ResolveOverrides::default(),
            env(&[("HI_PROVIDER", "xai")]),
        )
        .unwrap();
        assert_eq!(resolved.provider, ProviderName::Xai);
        assert_eq!(resolved.api_key, "");
    }

    #[test]
    fn partial_projection_ignores_unrelated_toml() {
        let file: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "main"
reasoning_effort = "high"
[sync]
enabled = true
[profiles.main]
provider = "ollama"
model = "qwen"
tool_mode = "read-only"
[profiles.main.extra]
anything = 42
"#,
        )
        .unwrap();
        assert_eq!(file.profiles["main"].model.as_deref(), Some("qwen"));
    }
}
