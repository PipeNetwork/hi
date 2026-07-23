use std::sync::Arc;

use anyhow::{Context, Result};
use hi_agent::{AgentConfig, AgentPaths, AgentRouting};
use hi_ai::{AnthropicProvider, OpenAiProvider, Provider};
use hi_shell::{ShellConfig, serve_stdio};

#[tokio::main]
async fn main() -> Result<()> {
    let provider_name = std::env::var("HI_PROVIDER").unwrap_or_else(|_| "openai".into());
    if provider_name != "openai" && provider_name != "anthropic" {
        anyhow::bail!("unsupported HI_PROVIDER {provider_name:?}; expected openai or anthropic");
    }
    let model = std::env::var("HI_MODEL").unwrap_or_else(|_| "grok-code-fast-1".into());
    let base_url = std::env::var("HI_BASE_URL").unwrap_or_else(|_| {
        if provider_name == "anthropic" {
            "https://api.anthropic.com".into()
        } else {
            "https://api.openai.com/v1".into()
        }
    });
    let api_key = match provider_name.as_str() {
        "anthropic" => std::env::var("HI_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
            .context("set HI_API_KEY or ANTHROPIC_API_KEY")?,
        "openai" => std::env::var("HI_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .context("set HI_API_KEY or OPENAI_API_KEY")?,
        _ => unreachable!("provider validated above"),
    };
    let provider: Arc<dyn Provider> = if provider_name == "anthropic" {
        Arc::new(AnthropicProvider::new(base_url, api_key))
    } else {
        Arc::new(OpenAiProvider::new(base_url, api_key))
    };
    let template = AgentConfig {
        paths: AgentPaths::default(),
        routing: AgentRouting {
            model,
            provider_route: Some(provider_name),
            ..AgentRouting::default()
        },
        ..AgentConfig::default()
    };

    let models = std::env::var("HI_MODELS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let local = tokio::task::LocalSet::new();
    local
        .run_until(serve_stdio(ShellConfig {
            provider,
            template,
            models,
        }))
        .await
}
