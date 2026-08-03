//! Headless independent skeptic review entrypoint (`--skeptic-review`).

use anyhow::{Context, Result};
use hi_agent::{Agent, AgentConfig, AgentRouting, AgentSubagents, SkepticVerdict};
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
        routing: AgentRouting {
            model: settings.model.clone(),
            compat: settings.compat,
            deepseek_compat: settings.deepseek_compat,
            ..AgentRouting::default()
        },
        // Reviewer model: HI_SKEPTIC_MODEL/profile, else fall back to --model.
        subagents: AgentSubagents {
            skeptic_model: Some(skeptic_model.unwrap_or_else(|| settings.model.clone())),
            ..AgentSubagents::default()
        },
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(provider, config).context("initializing reviewer runtime")?;
    // Transport Unavailable is an error for offline eval — never count as Approve.
    let verdict = agent
        .review_diff(&req.objective, &req.sub_goal, &req.diff)
        .await;
    let (objected, objections) = match verdict {
        SkepticVerdict::Object(objs) | SkepticVerdict::Escalate(objs) => (true, objs),
        SkepticVerdict::Approve => (false, Vec::new()),
        SkepticVerdict::Unavailable(reason) => {
            anyhow::bail!("skeptic review unavailable: {reason}");
        }
    };
    println!(
        "{}",
        serde_json::json!({ "objected": objected, "objections": objections })
    );
    Ok(())
}
