//! Headless independent skeptic review entrypoint (`--skeptic-review`).

use anyhow::{Context, Result};
use hi_agent::{Agent, AgentConfig};
use hi_ai::Provider;

use crate::config::Settings;

pub(crate) async fn run_skeptic_review(
    provider: std::sync::Arc<dyn Provider>,
    settings: &Settings,
    skeptic_model: Option<String>,
) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Req {
        objective: String,
        sub_goal: String,
        diff: String,
    }
    let input = std::io::read_to_string(std::io::stdin())
        .context("reading skeptic-review JSON from stdin")?;
    let req: Req =
        serde_json::from_str(&input).context("parsing skeptic-review JSON from stdin")?;
    let config = AgentConfig {
        model: settings.model.clone(),
        // Reviewer model: HI_SKEPTIC_MODEL/profile, else fall back to --model.
        skeptic_model: Some(skeptic_model.unwrap_or_else(|| settings.model.clone())),
        compat: settings.compat,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(provider, config).context("initializing reviewer runtime")?;
    let (objected, objections) = agent
        .review_diff(&req.objective, &req.sub_goal, &req.diff)
        .await;
    println!(
        "{}",
        serde_json::json!({ "objected": objected, "objections": objections })
    );
    Ok(())
}
