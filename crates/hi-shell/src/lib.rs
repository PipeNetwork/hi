//! ACP frontend for the `hi-agent` coding harness.

mod server;

use std::sync::Arc;

use anyhow::{Result, anyhow};
use hi_ai::{
    AnthropicProvider, McpDiscoveryProvider, OpenAiProvider, PipeMcpClient, Provider, TokenSource,
};
use hi_provider_config::{ProviderName, ResolvedProviderConfig};

pub use hi_provider_config::ResolvedProviderConfig as ProviderConfig;

pub use server::{HiShell, ShellConfig, serve_stdio};

// A free function, not an inherent impl: `ResolvedProviderConfig` is defined in
// `hi-provider-config`, and inherent impls on foreign types do not compile.
pub fn build_provider(config: &ResolvedProviderConfig) -> Result<Arc<dyn Provider>> {
    let provider: Arc<dyn Provider> = match config.provider {
        ProviderName::Anthropic => Arc::new(AnthropicProvider::new(
            config.base_url.clone(),
            config.api_key.clone(),
        )),
        ProviderName::Pipenetwork => {
            let inner =
                OpenAiProvider::new_pipenetwork(config.base_url.clone(), config.api_key.clone());
            if let Some(url) = &config.mcp_url {
                Arc::new(McpDiscoveryProvider::new(
                    Box::new(inner),
                    PipeMcpClient::new(url, config.api_key.clone()),
                ))
            } else {
                Arc::new(inner)
            }
        }
        ProviderName::Xai if config.api_key.is_empty() => {
            let source = hi_ai::xai_auth::XaiTokenSource::from_store()
                .ok_or_else(|| anyhow!("set HI_API_KEY or XAI_API_KEY, or sign in with hi"))?;
            Arc::new(OpenAiProvider::with_token_source(
                config.base_url.clone(),
                Arc::new(source) as Arc<dyn TokenSource>,
            ))
        }
        ProviderName::Openai | ProviderName::Xai | ProviderName::Ollama => Arc::new(
            OpenAiProvider::new(config.base_url.clone(), config.api_key.clone()),
        ),
    };
    Ok(provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    // hi-provider-config sits below hi-ai and cannot reference its constants;
    // this crate sees both, so it owns the tripwire keeping the duplicated
    // default in sync. A divergence here once sent the Pipe API key to a
    // different host than `hi` itself uses.
    #[test]
    fn pipe_mcp_default_matches_canonical_constant() {
        assert_eq!(
            ProviderName::Pipenetwork.default_mcp_url(),
            Some(hi_ai::PIPE_MCP_DEFAULT_URL)
        );
    }
}
