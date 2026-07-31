use anyhow::Result;
use hi_agent::{AgentConfig, AgentPaths, AgentRouting};
use hi_provider_config::{ResolveOverrides, load, resolve};
use hi_shell::{ShellConfig, serve_stdio};

#[tokio::main]
async fn main() -> Result<()> {
    let settings = resolve(&load(None)?, ResolveOverrides::default())?;
    let provider = hi_shell::build_provider(&settings)?;
    let template = AgentConfig {
        paths: AgentPaths::default(),
        routing: AgentRouting {
            model: settings.model,
            provider_route: Some(settings.provider.as_str().to_owned()),
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
