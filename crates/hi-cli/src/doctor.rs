//! CLI wiring for `hi doctor` / in-session `/doctor`.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;

use hi_agent::doctor::{
    Check, DoctorInput, DoctorReport, print_report, probe_mcp, render_report_text, run_doctor,
};

use crate::config::{self, Cli, Settings};
use crate::provider::provider_label;

/// `hi doctor [--json]` one-shot (argv after the `doctor` token).
pub async fn run_doctor_cli(args: &[String]) -> Result<()> {
    let json = args.iter().any(|a| a == "--json" || a == "-j");
    if args
        .iter()
        .any(|a| a == "--help" || a == "-h" || a == "help")
    {
        print_usage();
        return Ok(());
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let report = collect_report(&cwd, None, None).await;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into())
        );
    } else {
        print_report(&report);
    }
    if report.failing_count > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// In-session `/doctor` using already-resolved settings + optional live facts.
pub async fn run_doctor_for_session(
    cwd: &Path,
    settings: &Settings,
    session: SessionDoctorFacts<'_>,
) -> DoctorReport {
    collect_report(cwd, Some(settings), Some(session)).await
}

pub struct SessionDoctorFacts<'a> {
    pub model: &'a str,
    pub verify_summary: &'a str,
    pub lsp_summary: Option<&'a str>,
    pub checkpoint_count: usize,
    pub workspace_root: Option<&'a Path>,
}

pub fn report_text(report: &DoctorReport) -> String {
    render_report_text(report)
}

async fn collect_report(
    cwd: &Path,
    settings: Option<&Settings>,
    session: Option<SessionDoctorFacts<'_>>,
) -> DoctorReport {
    let project_config = {
        let p = cwd.join("hi.toml");
        p.is_file().then_some(p)
    };
    let user_config = config::default_config_path().filter(|p| p.is_file());

    let mut input = DoctorInput {
        cwd: cwd.to_path_buf(),
        workspace_root: session
            .as_ref()
            .and_then(|s| s.workspace_root.map(|p| p.to_path_buf())),
        project_config,
        user_config,
        ..DoctorInput::default()
    };

    if let Some(s) = session.as_ref() {
        input.verify_summary = Some(s.verify_summary.to_string());
        input.lsp_summary = s.lsp_summary.map(|s| s.to_string());
        input.checkpoint_count = Some(s.checkpoint_count);
        input.model = Some(s.model.to_string());
    }

    match settings {
        Some(settings) => fill_from_settings(&mut input, settings).await,
        None => {
            let cli = Cli::parse_from(["hi"]);
            match config::load_config(None) {
                Ok(file) => match config::resolve(&cli, &file) {
                    Ok(settings) => fill_from_settings(&mut input, &settings).await,
                    Err(err) => input.settings_error = Some(err.to_string()),
                },
                Err(err) => input.settings_error = Some(err.to_string()),
            }
        }
    }

    run_doctor(&input)
}

async fn fill_from_settings(input: &mut DoctorInput, settings: &Settings) {
    input.provider_label = Some(provider_label(settings.provider).to_string());
    if input.model.is_none() {
        input.model = Some(settings.model.clone());
    }
    input.base_url = Some(settings.base_url.clone());
    let key = settings.api_key.trim();
    if key.is_empty() {
        input.credentials_ok = false;
        input.credentials = Some("no API key resolved".into());
    } else {
        input.credentials_ok = true;
        input.credentials = Some(format!("api_key {}", config::mask_key(key)));
    }

    if let Some(url) = settings.mcp_url.as_deref() {
        input.mcp = Some(probe_mcp(url, &settings.api_key, &settings.model).await);
    } else {
        input.mcp = Some(Check::pass(
            "mcp endpoint",
            "not configured (optional for this provider)",
        ));
    }
}

fn print_usage() {
    println!(
        "\
hi doctor — diagnose common setup and runtime problems

Usage:
  hi doctor
  hi doctor --json

Checks config discovery, credentials, git/workspace, and (when configured) the
MCP endpoint. Exits 1 when any check fails.
"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cold_start_produces_report() {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let report = collect_report(&cwd, None, None).await;
        assert!(!report.checks.is_empty());
        assert!(report.checks.iter().any(|c| c.label == "git"));
        assert!(report.checks.iter().any(|c| c.label == "workspace"));
    }
}
