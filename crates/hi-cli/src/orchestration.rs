//! CLI orchestration helpers extracted from `main` so the binary stays a thin
//! dispatcher: best-of-N, sync config, MCP/HF side commands.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use hi_agent::VerifyStage;
use hi_ai::PipeMcpClient;

use crate::bestof;
use crate::config::{Cli, Config, Settings};
use crate::provider::provider_label;
use crate::report::pipeline_command;
use crate::session;
use crate::sync;

/// Run `--best-of N` in isolated worktrees; returns whether a candidate completed.
pub(crate) fn run_best_of(
    cli: &Cli,
    settings: &Settings,
    workspace_root: &Path,
    state_root: &Path,
    verify_stages: &[VerifyStage],
    quality_max_verify_repairs: u32,
    prompt: &str,
    report_path: Option<&Path>,
) -> Result<bool> {
    let Some(verify) = pipeline_command(verify_stages) else {
        anyhow::bail!("--best-of requires a resolved verification pipeline");
    };
    if !hi_tools::worktree::in_git_repo(workspace_root) {
        anyhow::bail!("--best-of requires a git repository");
    }
    let exe = std::env::current_exe().context("locating the hi executable")?;
    bestof::run(&bestof::BestOf {
        exe: &exe,
        provider: provider_label(settings.provider),
        model: &settings.model,
        base_url: &settings.base_url,
        api_key: &settings.api_key,
        verify: &verify,
        prompt,
        candidates: cli.best_of,
        max_steps: cli.max_steps,
        max_verify: quality_max_verify_repairs,
        workspace_root,
        state_root,
        report: report_path,
    })
}

/// Build remote-session sync credentials.
///
/// Precedence (first non-empty wins):
/// 1. `HI_SYNC_BASE_URL` / `HI_SYNC_API_KEY` env
/// 2. config file `[sync]` section
/// 3. CLI `--base-url` / `--api-key` when present
/// 4. resolved provider `settings` (profile defaults)
pub(crate) fn build_sync_config(
    settings: &Settings,
    cli: &Cli,
    file: &Config,
) -> sync::SyncConfig {
    let sync_section = file.sync.as_ref();
    let base_url = first_nonempty(&[
        std::env::var("HI_SYNC_BASE_URL").ok(),
        sync_section.and_then(|s| s.base_url.clone()),
        cli.base_url.clone(),
        Some(settings.base_url.clone()),
    ])
    .map(|u| u.trim_end_matches('/').to_string())
    .unwrap_or_default();
    let file_api_key = sync_section.and_then(|s| {
        s.api_key
            .clone()
            .filter(|k| !k.is_empty())
            .or_else(|| {
                s.api_key_env
                    .as_deref()
                    .and_then(|env_var| std::env::var(env_var).ok())
                    .filter(|k| !k.is_empty())
            })
    });
    let api_key = first_nonempty(&[
        std::env::var("HI_SYNC_API_KEY").ok(),
        file_api_key,
        cli.api_key.clone(),
        Some(settings.api_key.clone()),
    ])
    .unwrap_or_default();
    let machine_id = session::machine_id();
    let cwd_digest = Some(session::cwd_digest());
    sync::SyncConfig {
        base_url,
        api_key,
        machine_id,
        cwd_digest,
    }
}

/// First non-empty string in precedence order (env → file → cli → settings).
fn first_nonempty(candidates: &[Option<String>]) -> Option<String> {
    candidates
        .iter()
        .filter_map(|c| c.as_ref())
        .map(|s| s.trim())
        .find(|s| !s.is_empty())
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_nonempty_prefers_earlier_rungs() {
        assert_eq!(
            first_nonempty(&[
                None,
                Some(String::new()),
                Some("cli".into()),
                Some("settings".into()),
            ])
            .as_deref(),
            Some("cli")
        );
    }
}

pub(crate) async fn run_mcp_command(settings: &Settings) -> Result<()> {
    let Some(url) = settings.mcp_url.as_deref() else {
        return Err(anyhow!("no MCP URL configured for this provider"));
    };
    let report = mcp_inspect(url, &settings.api_key, &settings.model).await?;
    print!("{report}");
    Ok(())
}

pub(crate) async fn run_hf_cli(args: &[String]) -> Result<()> {
    if args.is_empty() {
        print!(
            "{}",
            hi_tools::handle_hf_command("help", &mut hi_tools::HfCommandState::default()).await?
        );
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("download")
        && args
            .get(2)
            .map(String::as_str)
            .is_some_and(|arg| matches!(arg, "--keep" | "keep"))
    {
        let repo = args
            .get(1)
            .ok_or_else(|| anyhow!("usage: hi hf download <repo[@revision]> --keep <dir>"))?;
        let dir = args
            .get(3)
            .ok_or_else(|| anyhow!("usage: hi hf download <repo[@revision]> --keep <dir>"))?;
        print!(
            "{}",
            hi_tools::download_repo_keep_foreground(repo, dir).await?
        );
        return Ok(());
    }

    let mut state = hi_tools::HfCommandState::default();
    match hi_tools::handle_hf_command_result(&args.join(" "), &mut state).await? {
        hi_tools::HfCommandResult::Text(text) => print!("{text}"),
        hi_tools::HfCommandResult::MlxReady(run) => print!("{}", run.message),
    }
    Ok(())
}

/// Build the MCP inspection report (server, tools, model count, current model)
/// as a plain-text block. Shared by the `hi mcp` one-shot and the REPL `/mcp`
/// command so their output can't drift.
pub(crate) async fn mcp_inspect(url: &str, api_key: &str, current_model: &str) -> Result<String> {
    let client = PipeMcpClient::new(url, api_key);
    let (server, protocol) = client.server_info().await?;
    let tools = client.tools_list().await?;
    let models = client.list_models().await?;
    let mut out = String::new();
    out.push_str(&format!("mcp_url:  {url}\n"));
    out.push_str(&format!("server:   {server}\n"));
    out.push_str(&format!("protocol: {protocol}\n"));
    out.push_str("tools:\n");
    for tool in tools {
        let title = tool.title.as_deref().unwrap_or("");
        if title.is_empty() {
            out.push_str(&format!("  {}\n", tool.name));
        } else {
            out.push_str(&format!("  {}  - {}\n", tool.name, title));
        }
    }
    out.push_str(&format!("models:   {}\n", models.len()));
    if let Some(model) = models.iter().find(|m| m.id == current_model) {
        let provider = model.provider_label.as_deref().unwrap_or("Pipe");
        out.push_str(&format!("current:  {} · {}\n", model.id, provider));
    }
    Ok(out)
}
