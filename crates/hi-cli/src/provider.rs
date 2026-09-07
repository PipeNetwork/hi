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

// `Provider` defaults are deliberately conservative, so transparent wrappers
// must forward both the exact declaration and every possible route member.
macro_rules! forward_provider_capabilities {
    ($self:ident, $inner:ident) => {
        fn capabilities(&$self) -> hi_ai::ProviderCapabilities {
            $self.$inner.capabilities()
        }

        fn capability_candidates(
            &$self,
            route: &str,
            model: &str,
        ) -> Vec<hi_ai::ProviderCapabilityCandidate> {
            $self.$inner.capability_candidates(route, model)
        }

        fn capability_candidates_for_request(
            &$self,
            route: &str,
            model: &str,
            context: hi_ai::ProviderRequestContext<'_>,
        ) -> Vec<hi_ai::ProviderCapabilityCandidate> {
            $self
                .$inner
                .capability_candidates_for_request(route, model, context)
        }
    };
}
pub(crate) use forward_provider_capabilities;

pub(crate) fn provider_label(provider: ProviderName) -> &'static str {
    // Same string as config files and `--provider` use, so a label can't drift
    // from the name a user is expected to type.
    provider.as_str()
}

/// Keep the provider family readable while fencing capability observations by
/// the concrete endpoint. Only a BLAKE3 digest of the endpoint enters session
/// state or diagnostics, so embedded credentials and query tokens stay out.
pub(crate) fn agent_provider_route(settings: &Settings) -> hi_agent::AgentProviderRoute {
    let label = provider_label(settings.provider);
    let base_url = resolved_credential_safe_base_url(&settings.base_url, settings.provider);
    let capability_label =
        capability_provider_label(settings.provider, &base_url, settings.deepseek_compat);
    provider_route_for_endpoint(
        label,
        capability_label,
        &base_url,
        settings.api_unix_socket.as_deref(),
    )
}

fn capability_provider_label(
    provider: ProviderName,
    base_url: &str,
    deepseek_compat: hi_ai::DeepSeekCompat,
) -> &'static str {
    if provider == ProviderName::Openai
        && deepseek_compat == hi_ai::DeepSeekCompat::Auto
        && hi_provider_config::is_official_deepseek_endpoint(base_url)
    {
        "deepseek"
    } else {
        provider_label(provider)
    }
}

fn provider_route_for_endpoint(
    label: &str,
    capability_label: &str,
    base_url: &str,
    api_unix_socket: Option<&std::path::Path>,
) -> hi_agent::AgentProviderRoute {
    let endpoint_identity = api_unix_socket.map_or(base_url.to_string(), |socket| {
        format!("{base_url}\0unix-socket\0{}", socket.to_string_lossy())
    });
    hi_agent::AgentProviderRoute::new(
        label,
        hi_ai::endpoint_capability_route(capability_label, &endpoint_identity),
    )
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
    let resolved = resolved_credential_safe_base_url(configured, provider);
    let trimmed = configured.trim();
    if !trimmed.is_empty() && !crate::orchestration::sync_base_url_is_safe(trimmed) {
        eprintln!(
            "warning: configured {} base_url is not https (or loopback http); \
             using the provider default '{resolved}' to avoid exposing credentials",
            provider_label(provider),
        );
    }
    resolved
}

fn resolved_credential_safe_base_url(configured: &str, provider: ProviderName) -> String {
    let trimmed = configured.trim();
    if trimmed.is_empty() || crate::orchestration::sync_base_url_is_safe(trimmed) {
        configured.to_string()
    } else {
        provider.default_base_url().to_string()
    }
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
            let auth = crate::x402::pipenetwork_token_source(settings);
            let mut provider = OpenAiProvider::new_pipenetwork_with_token_source(base_url, auth);
            if let Some(settler) = crate::x402::build_settler(settings) {
                provider = provider.with_x402(settler, settings.x402.max_usd);
            }
            Box::new(provider)
        } else {
            Box::new(OpenAiProvider::new(base_url, api_key.clone()))
        };
        if settings.provider == ProviderName::Pipenetwork
            && let Some(mcp_url) = settings.mcp_url.clone()
            && !api_key.trim().is_empty()
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
        capability_route: agent_provider_route(settings).capability_identity,
    }
}

/// The complete routing snapshot shared by startup and every frontend switch.
pub(crate) fn resolved_agent_routing(
    settings: &Settings,
    context_window: Option<u32>,
    max_tokens: u32,
    temperature: Option<f32>,
) -> hi_agent::AgentRouting {
    let route = agent_provider_route(settings);
    hi_agent::AgentRouting {
        model: settings.model.clone(),
        provider_route: Some(route.label),
        capability_route: Some(route.capability_identity),
        requested_max_tokens: settings.max_tokens,
        max_tokens,
        max_tokens_explicit: settings.max_tokens_explicit,
        temperature,
        top_p: settings.top_p,
        output_token_parameter: settings.output_token_parameter,
        thinking_budget: settings.thinking_budget,
        reasoning_effort: settings.reasoning_effort,
        tool_mode: settings.tool_mode,
        compat: settings.compat,
        deepseek_compat: settings.deepseek_compat,
        context_window,
    }
}

pub(crate) fn switched_routing(settings: &Settings) -> hi_agent::AgentRouting {
    resolved_agent_routing(
        settings,
        None,
        effective_max_tokens_for_model(settings, None),
        None,
    )
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
                || !settings.api_key.trim().is_empty()
                || settings.x402.enabled(),
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

#[derive(Clone, Debug, Default)]
pub(crate) struct LiveModelMetadata {
    pub(crate) context_window: Option<u32>,
    pub(crate) max_output_tokens: Option<u32>,
    pub(crate) price: Option<(f64, f64)>,
    /// Successful discovery evidence that can seed the request-time registry
    /// without causing a second provider call.
    pub(crate) provider_capabilities: Option<hi_ai::ProviderCapabilities>,
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
    let declared = provider.capabilities();
    match tokio::time::timeout(timeout, provider.list_models()).await {
        Ok(Ok(served)) => {
            let mut metadata = served
                .into_iter()
                .find(|m| m.id == model)
                .map(|m| live_model_metadata(m, declared))
                .unwrap_or_default();
            // `/models` came from one endpoint. It cannot safely override the
            // conservative intersection for a fallback/MoA route whose other
            // members may receive this exact request.
            if provider.capability_candidates("startup", model).len() != 1 {
                metadata.provider_capabilities = None;
            }
            metadata
        }
        Ok(Err(_)) | Err(_) => LiveModelMetadata::default(),
    }
}

fn live_model_metadata(
    model: hi_ai::ServedModel,
    mut capabilities: hi_ai::ProviderCapabilities,
) -> LiveModelMetadata {
    capabilities.request_limits.max_input_tokens = model.context_window;
    capabilities.request_limits.max_output_tokens = model.max_output_tokens;
    // `/models` identifiers are routable names, not immutable revision
    // evidence. Preserve a revision asserted by the provider capability
    // record, but never manufacture one from the requested/served model ID.
    for tag in &model.capabilities {
        match tag.trim().to_ascii_lowercase().as_str() {
            "tools" | "tool_calls" | "function_calling" => {
                capabilities.native_tool_calls = true;
                // A generic function-calling advertisement proves the model
                // can choose a tool automatically. It does not prove support
                // for an enforced `required` choice.
                capabilities.tool_choice.automatic = true;
            }
            "parallel_tool_calls" => capabilities.parallel_tool_calls = true,
            "structured_output" | "json_schema" => capabilities.structured_output = true,
            "vision" | "image_input" => capabilities.modalities.image_input = true,
            "audio_input" => capabilities.modalities.audio_input = true,
            "image_output" => capabilities.modalities.image_output = true,
            "audio_output" => capabilities.modalities.audio_output = true,
            _ => {}
        }
    }
    LiveModelMetadata {
        context_window: model.context_window,
        max_output_tokens: model.max_output_tokens,
        price: model.price,
        provider_capabilities: Some(capabilities),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MultiRouteMetadataProvider;

    #[async_trait::async_trait]
    impl hi_ai::Provider for MultiRouteMetadataProvider {
        async fn stream(
            &self,
            _: hi_ai::ChatRequest,
            _: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
        ) -> anyhow::Result<hi_ai::Completion> {
            unreachable!("metadata test never performs a chat request")
        }

        fn capability_candidates(
            &self,
            _: &str,
            model: &str,
        ) -> Vec<hi_ai::ProviderCapabilityCandidate> {
            ["primary", "fallback"]
                .into_iter()
                .map(|route| {
                    hi_ai::ProviderCapabilityCandidate::new(
                        hi_ai::CapabilityRoute::new(route, model),
                        hi_ai::ProviderCapabilities::default(),
                    )
                })
                .collect()
        }

        async fn list_models(&self) -> anyhow::Result<Vec<hi_ai::ServedModel>> {
            Ok(vec![hi_ai::ServedModel {
                id: "tool-model".into(),
                context_window: Some(8_192),
                max_output_tokens: None,
                price: None,
                provider_label: None,
                status: None,
                available: true,
                availability_reason: None,
                capabilities: vec!["tools".into()],
            }])
        }
    }

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

    #[test]
    fn capability_route_hashes_effective_endpoint_and_unix_socket_without_leaking_them() {
        let unsafe_configured = "http://user:secret@remote.invalid/v1?token=also-secret";
        let effective = resolved_credential_safe_base_url(unsafe_configured, ProviderName::Openai);
        assert_eq!(effective, ProviderName::Openai.default_base_url());

        let direct = provider_route_for_endpoint("openai", "openai", &effective, None);
        let socket_a = provider_route_for_endpoint(
            "openai",
            "openai",
            &effective,
            Some(std::path::Path::new("/private/run/provider-a.sock")),
        );
        let socket_b = provider_route_for_endpoint(
            "openai",
            "openai",
            &effective,
            Some(std::path::Path::new("/private/run/provider-b.sock")),
        );

        assert_eq!(direct.label, "openai");
        assert!(
            direct
                .capability_identity
                .starts_with("openai@endpoint:blake3:")
        );
        assert_ne!(direct.capability_identity, socket_a.capability_identity);
        assert_ne!(socket_a.capability_identity, socket_b.capability_identity);
        for identity in [direct, socket_a, socket_b] {
            assert!(!identity.capability_identity.contains("remote.invalid"));
            assert!(!identity.capability_identity.contains("secret"));
            assert!(!identity.capability_identity.contains("provider-a.sock"));
            assert!(!identity.capability_identity.contains("provider-b.sock"));
        }
    }

    #[test]
    fn official_deepseek_auto_route_keeps_openai_display_and_uses_deepseek_identity() {
        let endpoint = "https://api.deepseek.com/v1?token=also-secret";
        let auto_label =
            capability_provider_label(ProviderName::Openai, endpoint, hi_ai::DeepSeekCompat::Auto);
        let route = provider_route_for_endpoint("openai", auto_label, endpoint, None);

        assert_eq!(auto_label, "deepseek");
        assert_eq!(route.label, "openai");
        assert!(
            route
                .capability_identity
                .starts_with("deepseek@endpoint:blake3:")
        );
        assert!(!route.capability_identity.contains("api.deepseek.com"));
        assert!(!route.capability_identity.contains("secret"));
        assert_eq!(
            capability_provider_label(ProviderName::Openai, endpoint, hi_ai::DeepSeekCompat::Off,),
            "openai"
        );
        assert_eq!(
            capability_provider_label(ProviderName::Openai, endpoint, hi_ai::DeepSeekCompat::On,),
            "openai"
        );
        assert_eq!(
            capability_provider_label(
                ProviderName::Openai,
                "https://api.deepseek.com.example/v1",
                hi_ai::DeepSeekCompat::Auto,
            ),
            "openai"
        );
    }

    #[test]
    fn live_metadata_seeds_limits_without_inventing_a_revision() {
        let metadata = live_model_metadata(
            hi_ai::ServedModel {
                id: "model@revision-7".into(),
                context_window: Some(128_000),
                max_output_tokens: Some(16_384),
                price: None,
                provider_label: Some("test".into()),
                status: Some("available".into()),
                available: true,
                availability_reason: None,
                capabilities: vec!["parallel_tool_calls".into(), "vision".into()],
            },
            hi_ai::ProviderCapabilities::native_tools(true),
        );
        let capabilities = metadata.provider_capabilities.unwrap();
        assert_eq!(capabilities.actual_model_revision, None);
        assert_eq!(capabilities.request_limits.max_input_tokens, Some(128_000));
        assert_eq!(capabilities.request_limits.max_output_tokens, Some(16_384));
        assert!(capabilities.parallel_tool_calls);
        assert!(capabilities.modalities.image_input);
    }

    #[test]
    fn live_tool_metadata_enables_auto_for_a_conservative_single_route() {
        let metadata = live_model_metadata(
            hi_ai::ServedModel {
                id: "tool-model".into(),
                context_window: None,
                max_output_tokens: None,
                price: None,
                provider_label: None,
                status: None,
                available: true,
                availability_reason: None,
                capabilities: vec!["function_calling".into()],
            },
            hi_ai::ProviderCapabilities::default(),
        );
        let capabilities = metadata.provider_capabilities.unwrap();
        assert!(capabilities.native_tool_calls);
        assert!(capabilities.tool_choice.automatic);
        assert!(!capabilities.tool_choice.required);
    }

    #[tokio::test]
    async fn one_endpoint_metadata_cannot_override_a_multi_route_intersection() {
        let metadata = resolve_live_model_metadata_with_timeout(
            &MultiRouteMetadataProvider,
            "tool-model",
            std::time::Duration::from_secs(1),
        )
        .await;

        assert_eq!(metadata.context_window, Some(8_192));
        assert!(metadata.provider_capabilities.is_none());
    }
}

#[cfg(test)]
#[path = "provider_wrapper_tests.rs"]
mod wrapper_tests;
