//! Provider construction and labels shared by the CLI entrypoints.


use hi_ai::{
    AnthropicProvider, Backend, FallbackProvider, McpDiscoveryProvider, MoaProvider, OpenAiProvider,
    PipeMcpClient, Provider,
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
/// - **xAI** → latest Chat Completions Grok the client supports (`grok-4.5`),
///   not the session model. Weak/session coders on xAI were a common source of
///   empty or unparseable verdicts → `review unavailable`; a fixed strong
///   reviewer is better than disabling the gate. Override with `HI_SKEPTIC_MODEL`.
/// - **Elsewhere** → session model (same-model still catches concrete defects).
///
/// Review calls force temperature 0; verdict parsing tolerates preambles before
/// `APPROVE`/`OBJECT`.
pub(crate) fn default_skeptic_model(provider: ProviderName, session_model: &str) -> String {
    match provider {
        ProviderName::Pipenetwork => "pipe/glm-5.2".to_string(),
        // Chat Completions wire format (this client). Newer than the session
        // default `grok-4.3`; OpenAI adapter strips params 4.5 rejects.
        ProviderName::Xai => "grok-4.5".to_string(),
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

pub(crate) fn build_provider(settings: &Settings) -> Box<dyn Provider> {
    let base_url = settings.base_url.clone();
    let api_key = settings.api_key.clone();
    if settings.provider.is_anthropic() {
        Box::new(AnthropicProvider::new(base_url, api_key))
    } else {
        let inner: Box<dyn Provider> = if let Some(socket) = &settings.api_unix_socket {
            Box::new(OpenAiProvider::new_unix(base_url, api_key.clone(), socket))
        } else if settings.provider == ProviderName::Pipenetwork {
            Box::new(OpenAiProvider::new_pipenetwork(base_url, api_key.clone()))
        } else if let Some(source) = xai_oauth_token_source(settings.provider) {
            // Signed in with a grok.com subscription: the access token expires
            // in hours, so hand the provider a source that can re-mint it
            // rather than a fixed string that would strand a long session.
            Box::new(OpenAiProvider::with_token_source(base_url, source))
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

/// The primary backend, plus any fallbacks, as a single [`Provider`]. With no
/// fallbacks it's just the primary provider (no wrapper overhead).
pub(crate) fn build_chain(primary: &Settings, fallbacks: Vec<Settings>) -> Box<dyn Provider> {
    let passthrough: Box<dyn Provider> = if fallbacks.is_empty() {
        build_provider(primary)
    } else {
        let mut chain = vec![build_backend(primary)];
        chain.extend(fallbacks.iter().map(build_backend));
        Box::new(FallbackProvider::new(chain))
    };

    if !primary.moa.enabled {
        return passthrough;
    }

    Box::new(
        MoaProvider::new(passthrough, build_provider(primary), primary.moa.clone())
            .expect("MoA config should be validated before provider construction"),
    )
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LiveModelMetadata {
    pub(crate) context_window: Option<u32>,
    pub(crate) max_output_tokens: Option<u32>,
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

pub(crate) async fn resolve_live_model_metadata(provider: &dyn Provider, model: &str) -> LiveModelMetadata {
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

