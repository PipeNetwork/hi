//! Provider construction and labels shared by the CLI entrypoints.

use hi_ai::{
    AnthropicProvider, Backend, ConcurrencyLimitedProvider, DEFAULT_PROVIDER_REQUEST_CONCURRENCY,
    FallbackProvider, McpDiscoveryProvider, MoaProvider, OpenAiProvider, PipeMcpClient, Provider,
    ProviderConcurrencyConfig, XaiProvider,
};
use hi_routing::{
    Capability, CapabilitySet, HarnessDescriptor, ModelCandidate, RouteCandidate, RouteDecision,
    RouteRequirements, RouteResolver,
};

use crate::config::{ProviderName, Settings};

pub(crate) fn provider_label(provider: ProviderName) -> &'static str {
    // Same string as config files and `--provider` use, so a label can't drift
    // from the name a user is expected to type.
    provider.as_str()
}

/// The independent-review / `/goal team` skeptic model when neither
/// `HI_SKEPTIC_MODEL` nor the profile configures one.
///
/// - **Pipenetwork** → GLM-5.2 (second opinion, distinct from the coder route).
/// - **xAI** → grok-4.6 (Responses API), not the session model. Weak/session
///   coders on xAI were a common source of empty or unparseable verdicts →
///   `review unavailable`; a fixed strong reviewer is better than disabling
///   the gate. Override with `HI_SKEPTIC_MODEL`.
/// - **Elsewhere** → session model (same-model still catches concrete defects).
///
/// Review calls force temperature 0; verdict parsing tolerates preambles before
/// `APPROVE`/`OBJECT`.
pub(crate) fn default_skeptic_model(provider: ProviderName, session_model: &str) -> String {
    match provider {
        ProviderName::Pipenetwork => "pipe/glm-5.2".to_string(),
        ProviderName::Xai => "grok-4.6".to_string(),
        _ => session_model.to_string(),
    }
}

fn xai_oauth_token_source(
    provider: ProviderName,
) -> Option<std::sync::Arc<dyn hi_ai::TokenSource>> {
    if provider != ProviderName::Xai {
        return None;
    }
    hi_ai::xai_auth::XaiTokenSource::from_store()
        .map(|source| std::sync::Arc::new(source) as std::sync::Arc<dyn hi_ai::TokenSource>)
}

/// The base URL a provider may send the API key to. The key is attached to
/// every request, so the configured URL must not be able to redirect it onto
/// a plaintext or non-HTTP endpoint: only https (or loopback http for local
/// dev) is honored; anything else falls back to the provider's default
/// endpoint rather than leaking the credential. Same rule the sync path
/// applies (`sync_base_url_is_safe`). Pure so the policy is testable offline.
pub(crate) fn credential_safe_base_url(configured: &str, provider: ProviderName) -> String {
    let trimmed = configured.trim();
    if trimmed.is_empty() || crate::orchestration::sync_base_url_is_safe(trimmed) {
        return configured.to_string();
    }
    let fallback = provider.default_base_url().to_string();
    eprintln!(
        "warning: base_url '{trimmed}' is not https (or loopback http); \
         using the provider default '{fallback}' to avoid exposing the API key"
    );
    fallback
}

pub(crate) fn build_provider(settings: &Settings) -> Box<dyn Provider> {
    let base_url = credential_safe_base_url(&settings.base_url, settings.provider);
    let api_key = settings.api_key.clone();
    if settings.provider.is_anthropic() {
        Box::new(AnthropicProvider::new(base_url, api_key))
    } else if settings.provider == ProviderName::Xai {
        if let Some(source) = xai_oauth_token_source(settings.provider) {
            // Signed in with a grok.com subscription: the access token expires
            // in hours, so hand the provider a source that can re-mint it
            // rather than a fixed string that would strand a long session.
            Box::new(XaiProvider::with_token_source(base_url, source))
        } else {
            Box::new(XaiProvider::new(base_url, api_key))
        }
    } else {
        let inner: Box<dyn Provider> = if let Some(socket) = &settings.api_unix_socket {
            Box::new(OpenAiProvider::new_unix(base_url, api_key.clone(), socket))
        } else if settings.provider == ProviderName::Pipenetwork {
            Box::new(OpenAiProvider::new_pipenetwork(base_url, api_key.clone()))
        } else {
            Box::new(OpenAiProvider::new(base_url, api_key.clone()))
        };
        if settings.provider == ProviderName::Pipenetwork
            && let Some(mcp_url) = settings.mcp_url.clone()
        {
            Box::new(McpDiscoveryProvider::new(
                inner,
                PipeMcpClient::new(mcp_url, api_key),
            ))
        } else {
            inner
        }
    }
}

pub(crate) fn build_backend(settings: &Settings) -> Backend {
    Backend {
        provider: build_provider(settings),
        model: settings.model.clone(),
        label: format!("{}/{}", provider_label(settings.provider), settings.model),
    }
}

/// The primary backend, plus any fallbacks, as a single rate-bounded [`Provider`].
pub(crate) fn build_chain(primary: &Settings, fallbacks: Vec<Settings>) -> Box<dyn Provider> {
    let passthrough: Box<dyn Provider> = if fallbacks.is_empty() {
        build_provider(primary)
    } else {
        let mut chain = vec![build_backend(primary)];
        chain.extend(fallbacks.iter().map(build_backend));
        Box::new(FallbackProvider::new(chain).expect("chain is non-empty by construction"))
    };

    let composed: Box<dyn Provider> = if primary.moa.enabled {
        Box::new(
            MoaProvider::new(passthrough, build_provider(primary), primary.moa.clone())
                .expect("MoA config should be validated before provider construction"),
        )
    } else {
        passthrough
    };

    let concurrency = provider_concurrency_config();
    Box::new(
        ConcurrencyLimitedProvider::with_config(composed, concurrency)
            .expect("provider concurrency environment is normalized"),
    )
}

/// Describe the local hi harness and configured provider chain using the same
/// capability contract used by future harness adapters. Live `/models`
/// metadata can refine these static capabilities later; it must never weaken
/// the local tool/sandbox requirements.
pub(crate) fn resolve_startup_route(
    primary: &Settings,
    fallbacks: &[Settings],
    scope_id: impl Into<String>,
) -> Result<RouteDecision, hi_routing::RoutingError> {
    let local_harness = HarnessDescriptor {
        id: "hi".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        capabilities: CapabilitySet::default()
            .with(Capability::Streaming)
            .with(Capability::StructuredTools)
            .with(Capability::JsonSchema)
            .with(Capability::ToolReplay)
            .with(Capability::WorkspaceRead)
            .with(Capability::WorkspaceWrite)
            .with(Capability::ProcessExecution),
        isolation: "workspace".into(),
        network_allowed: false,
    };
    let mut settings = Vec::with_capacity(fallbacks.len() + 1);
    settings.push(primary);
    settings.extend(fallbacks.iter());
    let candidates = settings.into_iter().map(|settings| RouteCandidate {
        harness: local_harness.clone(),
        model: ModelCandidate {
            provider: provider_label(settings.provider).into(),
            model: settings.model.clone(),
            capabilities: CapabilitySet::default()
                .with(Capability::Streaming)
                .with(Capability::StructuredTools)
                .with(Capability::JsonSchema),
            available: true,
            credential_available: settings.provider == ProviderName::Ollama
                || !settings.api_key.trim().is_empty(),
            health: "configured".into(),
        },
    });
    RouteResolver::resolve(
        RouteRequirements {
            capabilities: CapabilitySet::default()
                .with(Capability::Streaming)
                .with(Capability::StructuredTools),
            require_available: true,
            require_credentials: true,
            scope_id: Some(scope_id.into()),
            policy_digest: None,
        },
        candidates,
    )
}

fn provider_concurrency_config() -> ProviderConcurrencyConfig {
    let max_concurrent = bounded_env_usize(
        "HI_PROVIDER_CONCURRENCY",
        DEFAULT_PROVIDER_REQUEST_CONCURRENCY,
        1,
        64,
    );
    let foreground_reserved = bounded_env_usize(
        "HI_PROVIDER_FOREGROUND_RESERVED",
        1,
        0,
        max_concurrent.saturating_sub(1),
    );
    let adaptive = std::env::var("HI_PROVIDER_ADAPTIVE_CONCURRENCY")
        .ok()
        .is_none_or(|value| !matches!(value.trim(), "0" | "false" | "off"));
    ProviderConcurrencyConfig {
        max_concurrent,
        foreground_reserved,
        adaptive,
    }
}

fn bounded_env_usize(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(min, max.max(min))
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LiveModelMetadata {
    pub(crate) context_window: Option<u32>,
    pub(crate) max_output_tokens: Option<u32>,
}

/// Metadata used while preparing startup. Live discovery is deliberately not
/// polled here because it is optional tuning and may hang indefinitely.
pub(crate) fn startup_live_model_metadata() -> LiveModelMetadata {
    LiveModelMetadata::default()
}

pub(crate) fn effective_max_tokens_for_model(
    settings: &Settings,
    advertised_max_output_tokens: Option<u32>,
) -> u32 {
    hi_ai::effective_coding_agent_max_tokens(
        &settings.model,
        settings.max_tokens,
        settings.max_tokens_explicit,
        advertised_max_output_tokens,
    )
}

pub(crate) async fn resolve_live_model_metadata(
    provider: &dyn Provider,
    model: &str,
) -> LiveModelMetadata {
    // Live metadata only tunes context/output limits; it must never hold the
    // interactive UI hostage when a provider's optional `/models` route hangs.
    // Continue with conservative defaults on timeout just as we do on errors.
    const STARTUP_METADATA_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
    resolve_live_model_metadata_with_timeout(provider, model, STARTUP_METADATA_TIMEOUT).await
}

pub(crate) async fn resolve_live_model_metadata_with_timeout(
    provider: &dyn Provider,
    model: &str,
    timeout: std::time::Duration,
) -> LiveModelMetadata {
    match tokio::time::timeout(timeout, provider.list_models()).await {
        Ok(Ok(served)) => served
            .into_iter()
            .find(|m| m.id == model)
            .map(|m| LiveModelMetadata {
                context_window: m.context_window,
                max_output_tokens: m.max_output_tokens,
            })
            .unwrap_or(LiveModelMetadata {
                context_window: None,
                max_output_tokens: None,
            }),
        Ok(Err(_)) | Err(_) => LiveModelMetadata {
            context_window: None,
            max_output_tokens: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_safe_base_url_keeps_https_and_loopback() {
        assert_eq!(
            credential_safe_base_url("https://api.x.ai/v1", ProviderName::Xai),
            "https://api.x.ai/v1"
        );
        assert_eq!(
            credential_safe_base_url("http://localhost:11434/v1", ProviderName::Ollama),
            "http://localhost:11434/v1"
        );
        assert_eq!(
            credential_safe_base_url("http://127.0.0.1:8080/v1", ProviderName::Openai),
            "http://127.0.0.1:8080/v1"
        );
    }

    #[test]
    fn credential_safe_base_url_falls_back_on_plaintext_remote() {
        // A plaintext-remote or non-HTTP endpoint must never receive the key:
        // the provider's (https) default is used instead.
        assert_eq!(
            credential_safe_base_url("http://evil.example/v1", ProviderName::Xai),
            "https://api.x.ai/v1"
        );
        assert_eq!(
            credential_safe_base_url("ftp://example.com", ProviderName::Anthropic),
            hi_provider_config::ProviderName::Anthropic.default_base_url()
        );
        assert_eq!(
            credential_safe_base_url("http://169.254.169.254/latest", ProviderName::Openai),
            "https://openrouter.ai/api/v1"
        );
    }
}
