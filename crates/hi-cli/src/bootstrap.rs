//! Early CLI bootstrap: flag validation, config load, and short-circuit commands.
//!
//! Keeps `main::run` focused on wiring the agent and choosing a run mode
//! (one-shot / TUI / REPL / daemon) rather than front-loading every preflight.

use anyhow::Result;
use clap::Parser;

use crate::config::{self, Cli, ProviderName, RsiRequested};
use crate::provider::{
    LiveModelMetadata, build_provider, effective_max_tokens_for_model, provider_label,
    resolve_live_model_metadata,
};
use crate::session;

/// Parse argv and reject incompatible flag combinations (exits process on error).
pub(crate) fn parse_and_validate_cli() -> Cli {
    let cli = Cli::parse();
    if let Some(id) = cli.sync_session_id.as_deref()
        && let Err(err) = crate::sync::validate_session_id(id)
    {
        eprintln!("{err}");
        std::process::exit(2);
    }
    if let Some(id) = cli.attach.as_deref()
        && let Err(err) = crate::sync::validate_session_id(id)
    {
        eprintln!("{err}");
        std::process::exit(2);
    }
    if cli.resume_local && cli.attach.is_none() {
        eprintln!("--resume-local requires --attach <SESSION_ID>");
        std::process::exit(2);
    }
    if cli.attach.is_some() && cli.daemon {
        eprintln!("--attach and --daemon cannot be used together");
        std::process::exit(2);
    }
    cli
}

/// Handle `--show-config` / `--list-sessions` before the heavy agent path.
///
/// Returns `Some(result)` when the process should exit after the short-circuit.
pub(crate) async fn maybe_short_circuit(cli: &Cli) -> Option<Result<()>> {
    if cli.show_config {
        return Some(print_show_config(cli).await);
    }
    if cli.list_sessions {
        return Some(session::list_sessions());
    }
    None
}

async fn print_show_config(cli: &Cli) -> Result<()> {
    let file = match config::load_config(cli.config.as_deref()) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("{err:#}");
            std::process::exit(2);
        }
    };
    match config::resolve(cli, &file) {
        Ok(settings) => {
            let live = if settings.provider == ProviderName::Pipenetwork {
                let provider = build_provider(&settings);
                resolve_live_model_metadata(provider.as_ref(), &settings.model).await
            } else {
                LiveModelMetadata {
                    context_window: None,
                    max_output_tokens: None,
                }
            };
            let effective_max_tokens =
                effective_max_tokens_for_model(&settings, live.max_output_tokens);
            println!("provider:   {}", provider_label(settings.provider));
            println!("model:      {}", settings.model);
            println!("base_url:   {}", settings.base_url);
            if let Some(mcp_url) = &settings.mcp_url {
                println!("mcp_url:    {mcp_url}");
            }
            println!("max_tokens: {}", effective_max_tokens);
            if let Some(limit) = live.max_output_tokens {
                println!("model_max_output_tokens: {limit}");
            }
            println!(
                "thinking:   {}",
                settings
                    .thinking_budget
                    .map(|b| b.to_string())
                    .unwrap_or_else(|| "off".into())
            );
            println!(
                "reasoning:  {}",
                settings
                    .reasoning_effort
                    .map(|e| e.as_str().to_string())
                    .unwrap_or_else(|| "off".into())
            );
            println!("tool_mode:  {:?}", settings.tool_mode);
            let rsi = config::resolve_rsi(cli, &file)?;
            println!("rsi_requested: {rsi:?}");
            println!(
                "rsi_active:    {}",
                if rsi == RsiRequested::Off {
                    "off"
                } else {
                    "on"
                }
            );
            println!("rsi_latest_turn_fully_observed: none");
            println!("compat:     {:?}", settings.compat);
            println!("api_key:    {}", config::mask_key(&settings.api_key));
            Ok(())
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    }
}
