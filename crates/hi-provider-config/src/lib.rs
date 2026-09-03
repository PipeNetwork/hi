//! Shared, provider-only projection of `hi` configuration.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    str::FromStr,
};

use serde::Deserialize;
use thiserror::Error;
use url::Url;

pub const DEFAULT_MAX_TOKENS: u32 = 8192;
pub const PIPE_DEEPSEEK_V4_FLASH_0731_MODEL_ID: &str = "pipe/deepseek-v4-flash-0731";
pub const PIPE_DEEPSEEK_V4_FLASH_VISION_EXP_MODEL_ID: &str = "pipe/deepseek-v4-flash-vision-exp";

/// Previous hi defaults. Resolve them to the production Flash SKU that has
/// both owned-RTX capacity and hosted overflow, rather than the experimental
/// vision route whose only live leaf may be unavailable.
pub fn remap_stale_pipenetwork_default_model(model: &str) -> Option<&'static str> {
    match model.trim() {
        "ipop/coder-balanced" | "pipe/auto-coder" => Some(PIPE_DEEPSEEK_V4_FLASH_0731_MODEL_ID),
        _ => None,
    }
}

pub fn apply_stale_pipenetwork_default_model(provider: ProviderName, model: &mut String) {
    if provider != ProviderName::Pipenetwork {
        return;
    }
    if let Some(remapped) = remap_stale_pipenetwork_default_model(model) {
        *model = remapped.to_string();
    }
}

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
            Self::Pipenetwork => Some(PIPE_DEEPSEEK_V4_FLASH_0731_MODEL_ID),
            Self::Anthropic => Some("claude-opus-4-8"),
            Self::Xai => Some("grok-4.6"),
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

/// Whether `base_url` has the same authenticated origin as the provider's
/// built-in endpoint. Paths may vary, but scheme, host, and effective port
/// must match and URL userinfo is never accepted.
pub fn is_official_provider_endpoint(provider: ProviderName, base_url: &str) -> bool {
    same_endpoint_origin(base_url, provider.default_base_url())
}

/// Whether a credential-bearing auxiliary provider endpoint (currently MCP)
/// stays on either the provider's official API or official MCP origin.
pub fn is_official_provider_service_endpoint(provider: ProviderName, url: &str) -> bool {
    is_official_provider_endpoint(provider, url)
        || provider
            .default_mcp_url()
            .is_some_and(|official| same_endpoint_origin(url, official))
}

/// DeepSeek is an official alternate endpoint for the OpenAI-compatible wire
/// format, but its credentials are distinct from OpenRouter/OpenAI credentials.
pub fn is_official_deepseek_endpoint(base_url: &str) -> bool {
    same_endpoint_origin(base_url, "https://api.deepseek.com")
}

/// Whether an endpoint is confined to this machine. Project-local provider
/// routes may use loopback without a durable folder-trust grant; every other
/// origin can disclose prompts or repository data off-device.
pub fn is_loopback_endpoint(endpoint: &str) -> bool {
    let Ok(url) = Url::parse(endpoint) else {
        return false;
    };
    if !url.username().is_empty() || url.password().is_some() {
        return false;
    }
    match url.host() {
        Some(url::Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(address)) => address.is_loopback(),
        Some(url::Host::Ipv6(address)) => address.is_loopback(),
        None => false,
    }
}

/// Compare authenticated URL origins without following path/query differences.
/// Userinfo is rejected because it can make log/display semantics ambiguous.
pub fn same_endpoint_origin(candidate: &str, trusted: &str) -> bool {
    let (Ok(candidate), Ok(trusted)) = (Url::parse(candidate), Url::parse(trusted)) else {
        return false;
    };
    candidate.username().is_empty()
        && candidate.password().is_none()
        && trusted.username().is_empty()
        && trusted.password().is_none()
        && candidate.scheme() == trusted.scheme()
        && candidate
            .host_str()
            .zip(trusted.host_str())
            .is_some_and(|(candidate, trusted)| candidate.eq_ignore_ascii_case(trusted))
        && candidate.port_or_known_default() == trusted.port_or_known_default()
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
    /// Runtime-only provenance for profiles loaded from an automatically
    /// merged repository `hi.toml`.
    #[serde(skip)]
    #[doc(hidden)]
    pub project_local: bool,
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
    #[error(
        "no DeepSeek API key configured; set DEEPSEEK_API_KEY, pair api_key/api_key_env with \
         the selected profile, or provide an explicit API key override"
    )]
    MissingDeepSeekApiKey,
    #[error(
        "custom endpoint has no explicitly paired API key; set api_key/api_key_env on its \
         profile or provide an explicit API key override (ambient stored/provider credentials \
         are only used for official provider endpoints)"
    )]
    UnpairedCustomEndpoint,
    #[error(
        "project-local profiles cannot read api_key_env '{0}' from the user's environment; \
         use a non-secret literal api_key, a trusted global profile, or an explicit API key override"
    )]
    ProjectLocalApiKeyEnv(String),
    #[error(
        "project-local remote provider routes are not trusted by this frontend; use a \
         loopback endpoint or an explicit configuration path/route override"
    )]
    UntrustedProjectCustomEndpoint,
    #[error(
        "custom MCP endpoint has no credential explicitly paired with that route; use an \
         official provider MCP/API origin, set mcp_url with api_key in a trusted profile, \
         use a project-local literal api_key plus custom base_url, or provide explicit \
         MCP URL and API key overrides together"
    )]
    UnpairedCustomMcpEndpoint,
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
        merge_project_config(&mut config, load_file(&local)?);
    }
    Ok(config)
}

fn merge_project_config(base: &mut ProviderConfigFile, mut project: ProviderConfigFile) {
    for profile in project.profiles.values_mut() {
        profile.project_local = true;
    }
    if project.default_profile.is_some() {
        base.default_profile = project.default_profile;
    }
    base.profiles.extend(project.profiles);
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
    let explicit_provider = explicit.provider;
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

    let provider = match explicit_provider.or_else(|| profile.and_then(|value| value.provider)) {
        Some(provider) => provider,
        // A typo in HI_PROVIDER must fail loudly: silently falling back to the
        // default provider would send whatever API key resolves to a host the
        // user never chose.
        None => match env("HI_PROVIDER") {
            Some(value) => value.parse()?,
            None => ProviderName::Openai,
        },
    };
    // A provider override changes the whole remote route, not just its label.
    // In particular, a default xAI profile's endpoint and literal key must not
    // follow an explicit Pipenetwork override. Profiles without a provider are
    // OpenAI profiles by the same default used above.
    let route_profile = match explicit_provider {
        Some(explicit_provider)
            if profile.is_some_and(|profile| {
                profile.provider.unwrap_or(ProviderName::Openai) != explicit_provider
            }) =>
        {
            None
        }
        _ => profile,
    };
    let explicit_api_key = explicit.api_key;
    if explicit_api_key.is_none()
        && let Some(env_name) = route_profile
            .filter(|profile| profile.project_local)
            .and_then(|profile| profile.api_key_env.as_ref())
    {
        return Err(ConfigError::ProjectLocalApiKeyEnv(env_name.clone()));
    }
    let project_profile_selects_route =
        explicit.base_url.is_none() && route_profile.is_some_and(|profile| profile.project_local);
    let profile_selects_base_url = explicit.base_url.is_none()
        && route_profile.is_some_and(|profile| profile.base_url.is_some());
    let mut model = explicit
        .model
        .or_else(|| route_profile.and_then(|value| value.model.clone()))
        .or_else(|| env("HI_MODEL"))
        .or_else(|| provider.default_model().map(str::to_owned))
        .ok_or(ConfigError::MissingModel)?;
    apply_stale_pipenetwork_default_model(provider, &mut model);
    let base_url = explicit
        .base_url
        .or_else(|| route_profile.and_then(|value| value.base_url.clone()))
        .or_else(|| env("HI_BASE_URL"))
        .unwrap_or_else(|| provider.default_base_url().to_owned());
    if project_profile_selects_route && !is_loopback_endpoint(&base_url) {
        return Err(ConfigError::UntrustedProjectCustomEndpoint);
    }
    #[derive(Clone, Copy)]
    enum McpSource {
        Explicit,
        Profile,
        Environment,
        Default,
    }
    let selected_mcp = explicit
        .mcp_url
        .map(|url| (url, McpSource::Explicit))
        .or_else(|| {
            route_profile
                .and_then(|value| value.mcp_url.clone())
                .map(|url| (url, McpSource::Profile))
        })
        .or_else(|| env("HI_MCP_URL").map(|url| (url, McpSource::Environment)))
        .or_else(|| {
            provider
                .default_mcp_url()
                .map(|url| (url.to_owned(), McpSource::Default))
        });
    let mcp_url = match selected_mcp {
        None => None,
        Some((url, _)) if url.trim().is_empty() => None,
        Some((url, _)) if is_official_provider_service_endpoint(provider, &url) => Some(url),
        Some((url, source)) => {
            let paired = match source {
                McpSource::Explicit => explicit_api_key.is_some(),
                McpSource::Profile if explicit_api_key.is_none() => {
                    route_profile.is_some_and(|profile| {
                        if profile.project_local {
                            profile.api_key.is_some()
                                && profile.base_url.as_deref().is_some_and(|url| {
                                    !is_official_provider_endpoint(provider, url)
                                })
                        } else {
                            profile.api_key.is_some() || profile.api_key_env.is_some()
                        }
                    })
                }
                McpSource::Profile | McpSource::Environment | McpSource::Default => false,
            };
            if !paired {
                return Err(ConfigError::UnpairedCustomMcpEndpoint);
            }
            Some(url)
        }
    };
    let profile_api_key = route_profile
        .and_then(|value| value.api_key.clone())
        .or_else(|| {
            let reference = route_profile.and_then(|value| value.api_key_env.as_deref())?;
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
        });
    let official_provider_endpoint = is_official_provider_endpoint(provider, &base_url);
    let official_deepseek_endpoint =
        provider == ProviderName::Openai && is_official_deepseek_endpoint(&base_url);
    let allow_unpaired_generic_key = !profile_selects_base_url;
    let ambient_api_key = if official_deepseek_endpoint {
        env("DEEPSEEK_API_KEY").or_else(|| {
            allow_unpaired_generic_key
                .then(|| env("HI_API_KEY"))
                .flatten()
        })
    } else if official_provider_endpoint {
        provider.key_envs().iter().find_map(|name| env(name))
    } else if allow_unpaired_generic_key {
        // `HI_API_KEY` is endpoint-generic, so preserve deliberate
        // HI_BASE_URL/HI_API_KEY or explicit-base-url workflows. Provider-
        // specific variables remain bound to their official origin.
        env("HI_API_KEY")
    } else {
        None
    };
    let api_key = explicit_api_key
        .or(profile_api_key)
        .or(ambient_api_key)
        // Ollama needs no key. xAI may authenticate through the signed-in
        // token store only at its official origin; an empty key at a custom
        // origin would make the builder forward that stored OAuth token.
        .or_else(|| {
            (matches!(provider, ProviderName::Ollama)
                || (provider == ProviderName::Xai && official_provider_endpoint))
                .then(String::new)
        })
        .ok_or({
            if official_deepseek_endpoint {
                ConfigError::MissingDeepSeekApiKey
            } else if official_provider_endpoint {
                ConfigError::MissingApiKey
            } else {
                ConfigError::UnpairedCustomEndpoint
            }
        })?;
    let max_tokens = explicit
        .max_tokens
        .or_else(|| route_profile.and_then(|value| value.max_tokens))
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
    fn provider_override_drops_mismatched_profile_route_and_credentials() {
        let file: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "work"
[profiles.work]
provider = "xai"
model = "grok-profile"
base_url = "https://xai-profile.invalid/v1"
mcp_url = "https://xai-profile.invalid/mcp"
api_key = "xai-profile-key"
max_tokens = 1234
"#,
        )
        .unwrap();
        let resolved = resolve_with_env(
            &file,
            ResolveOverrides {
                provider: Some(ProviderName::Pipenetwork),
                ..Default::default()
            },
            env(&[("PIPENETWORK_API_KEY", "pipe-key")]),
        )
        .unwrap();

        assert_eq!(resolved.provider, ProviderName::Pipenetwork);
        assert_eq!(resolved.model, PIPE_DEEPSEEK_V4_FLASH_0731_MODEL_ID);
        assert_eq!(
            resolved.base_url,
            ProviderName::Pipenetwork.default_base_url()
        );
        assert_eq!(
            resolved.mcp_url.as_deref(),
            ProviderName::Pipenetwork.default_mcp_url()
        );
        assert_eq!(resolved.api_key, "pipe-key");
        assert_eq!(resolved.max_tokens, DEFAULT_MAX_TOKENS);
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
    fn profile_custom_endpoint_cannot_consume_unpaired_ambient_key() {
        let file: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "project"
[profiles.project]
provider = "openai"
model = "attacker/model"
base_url = "https://attacker.example/v1"
"#,
        )
        .unwrap();

        let result = resolve_with_env(
            &file,
            ResolveOverrides::default(),
            env(&[
                ("HI_API_KEY", "global-generic-key"),
                ("OPENROUTER_API_KEY", "global-openrouter-key"),
            ]),
        );
        assert!(matches!(result, Err(ConfigError::UnpairedCustomEndpoint)));
    }

    #[test]
    fn project_profile_cannot_name_an_environment_credential_even_on_official_origin() {
        let mut global: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "global"
[profiles.global]
provider = "openai"
model = "global/model"
"#,
        )
        .unwrap();
        let project: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "project"
[profiles.project]
provider = "openai"
model = "project/model"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
"#,
        )
        .unwrap();
        merge_project_config(&mut global, project);

        let result = resolve_with_env(
            &global,
            ResolveOverrides::default(),
            env(&[("OPENROUTER_API_KEY", "user-secret")]),
        );
        assert!(matches!(
            result,
            Err(ConfigError::ProjectLocalApiKeyEnv(name)) if name == "OPENROUTER_API_KEY"
        ));

        let resolved = resolve_with_env(
            &global,
            ResolveOverrides {
                api_key: Some("explicit-key".into()),
                base_url: Some("https://openrouter.ai/api/v1".into()),
                ..Default::default()
            },
            env(&[("OPENROUTER_API_KEY", "user-secret")]),
        )
        .unwrap();
        assert_eq!(resolved.api_key, "explicit-key");
    }

    #[test]
    fn project_profile_literal_key_cannot_authorize_untrusted_custom_route() {
        let mut config = ProviderConfigFile::default();
        let project: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "project"
[profiles.project]
provider = "openai"
model = "project/model"
base_url = "https://project-gateway.example/v1"
api_key = "repository-test-key"
"#,
        )
        .unwrap();
        merge_project_config(&mut config, project);
        let result = resolve_with_env(&config, ResolveOverrides::default(), env(&[]));
        assert!(matches!(
            result,
            Err(ConfigError::UntrustedProjectCustomEndpoint)
        ));

        // An explicit route + key override is an operator decision and does
        // not consume the project's endpoint or credential.
        let resolved = resolve_with_env(
            &config,
            ResolveOverrides {
                base_url: Some("https://operator-gateway.example/v1".into()),
                api_key: Some("operator-key".into()),
                ..Default::default()
            },
            env(&[]),
        )
        .unwrap();
        assert_eq!(resolved.api_key, "operator-key");
    }

    #[test]
    fn project_profile_cannot_silently_select_official_remote_route() {
        let mut config = ProviderConfigFile::default();
        let project: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "project"
[profiles.project]
provider = "anthropic"
model = "claude-project"
api_key = "repository-owned-key"
"#,
        )
        .unwrap();
        merge_project_config(&mut config, project);

        let result = resolve_with_env(&config, ResolveOverrides::default(), env(&[]));
        assert!(matches!(
            result,
            Err(ConfigError::UntrustedProjectCustomEndpoint)
        ));

        let local: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "local"
[profiles.local]
provider = "openai"
model = "local-model"
base_url = "http://127.0.0.1:8080/v1"
api_key = "local"
"#,
        )
        .unwrap();
        let mut config = ProviderConfigFile::default();
        merge_project_config(&mut config, local);
        let resolved = resolve_with_env(&config, ResolveOverrides::default(), env(&[])).unwrap();
        assert_eq!(resolved.base_url, "http://127.0.0.1:8080/v1");
    }

    #[test]
    fn project_custom_mcp_cannot_receive_ambient_provider_key() {
        let mut config = ProviderConfigFile::default();
        let project: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "project"
[profiles.project]
provider = "pipenetwork"
model = "pipe/model"
base_url = "https://api.pipenetwork.ai/v1"
mcp_url = "https://attacker.example/mcp"
"#,
        )
        .unwrap();
        merge_project_config(&mut config, project);
        let result = resolve_with_env(
            &config,
            ResolveOverrides::default(),
            env(&[("PIPENETWORK_API_KEY", "ambient-pipe-key")]),
        );
        assert!(matches!(
            result,
            Err(ConfigError::UntrustedProjectCustomEndpoint)
        ));
    }

    #[test]
    fn project_custom_mcp_and_api_route_remain_blocked_without_trust_support() {
        let mut config = ProviderConfigFile::default();
        let project: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "project"
[profiles.project]
provider = "pipenetwork"
model = "pipe/model"
base_url = "https://project-api.example/v1"
mcp_url = "https://project-mcp.example/mcp"
api_key = "repository-test-key"
"#,
        )
        .unwrap();
        merge_project_config(&mut config, project);
        let result = resolve_with_env(&config, ResolveOverrides::default(), env(&[]));
        assert!(matches!(
            result,
            Err(ConfigError::UntrustedProjectCustomEndpoint)
        ));
    }

    #[test]
    fn custom_endpoint_accepts_profile_paired_or_explicit_generic_key() {
        let paired: ProviderConfigFile = toml::from_str(
            r#"
default_profile = "custom"
[profiles.custom]
provider = "openai"
model = "custom/model"
base_url = "https://gateway.example/v1"
api_key_env = "GATEWAY_KEY"
"#,
        )
        .unwrap();
        let resolved = resolve_with_env(
            &paired,
            ResolveOverrides::default(),
            env(&[("GATEWAY_KEY", "paired-key")]),
        )
        .unwrap();
        assert_eq!(resolved.api_key, "paired-key");

        let resolved = resolve_with_env(
            &ProviderConfigFile::default(),
            ResolveOverrides {
                model: Some("custom/model".into()),
                base_url: Some("https://gateway.example/v1".into()),
                ..Default::default()
            },
            env(&[("HI_API_KEY", "deliberate-generic-key")]),
        )
        .unwrap();
        assert_eq!(resolved.api_key, "deliberate-generic-key");
    }

    #[test]
    fn official_endpoint_matching_is_origin_strict() {
        assert!(is_official_provider_endpoint(
            ProviderName::Openai,
            "https://openrouter.ai:443/another/path"
        ));
        assert!(!is_official_provider_endpoint(
            ProviderName::Openai,
            "http://openrouter.ai/api/v1"
        ));
        assert!(!is_official_provider_endpoint(
            ProviderName::Openai,
            "https://openrouter.ai:444/api/v1"
        ));
        assert!(!is_official_provider_endpoint(
            ProviderName::Openai,
            "https://openrouter.ai@attacker.example/api/v1"
        ));
        assert!(is_official_deepseek_endpoint("https://api.deepseek.com/v1"));
        assert!(!is_official_deepseek_endpoint("http://api.deepseek.com/v1"));
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

    #[test]
    fn pipenetwork_default_and_stale_aliases_resolve_to_flash_0731() {
        assert_eq!(
            ProviderName::Pipenetwork.default_model(),
            Some(PIPE_DEEPSEEK_V4_FLASH_0731_MODEL_ID)
        );
        assert_eq!(
            remap_stale_pipenetwork_default_model("ipop/coder-balanced"),
            Some(PIPE_DEEPSEEK_V4_FLASH_0731_MODEL_ID)
        );
        assert_eq!(
            remap_stale_pipenetwork_default_model("pipe/auto-coder"),
            Some(PIPE_DEEPSEEK_V4_FLASH_0731_MODEL_ID)
        );
        assert_eq!(remap_stale_pipenetwork_default_model("pipe/glm-5.2"), None);
        assert_eq!(
            remap_stale_pipenetwork_default_model(PIPE_DEEPSEEK_V4_FLASH_0731_MODEL_ID),
            None
        );

        let resolved = resolve_with_env(
            &ProviderConfigFile::default(),
            ResolveOverrides {
                provider: Some(ProviderName::Pipenetwork),
                api_key: Some("key".into()),
                model: Some("ipop/coder-balanced".into()),
                ..Default::default()
            },
            env(&[]),
        )
        .unwrap();
        assert_eq!(resolved.model, PIPE_DEEPSEEK_V4_FLASH_0731_MODEL_ID);
    }
}
