//! CLI wiring for `hi doctor` / in-session `/doctor`.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use clap::Parser;

use hi_agent::doctor::{
    Check, DoctorInput, DoctorReport, print_report, probe_mcp, render_report_text, run_doctor,
};

use crate::config::{self, Cli, Settings};
use crate::provider::provider_label;
use crate::sync_store::{SyncStatus, SyncStore};

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

    input.runtime_checks = local_runtime_checks(None).await;

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

async fn local_runtime_checks(session_id: Option<&str>) -> Vec<Check> {
    let mut checks = vec![Check::pass(
        "client version",
        format!("hi {}", env!("CARGO_PKG_VERSION")),
    )];
    let runtime_dir = crate::local_runtime::runtime_dir();
    // Runtimes are keyed by session id, so probing one hardcoded name would
    // never observe a real leader; inspect every socket actually present.
    let mut runtime_sessions: Vec<String> = match session_id {
        Some(id) => vec![id.to_string()],
        None => std::fs::read_dir(&runtime_dir)
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .filter_map(|entry| {
                        let name = entry.file_name().to_string_lossy().into_owned();
                        name.strip_suffix(".sock").map(str::to_string)
                    })
                    // Stray files that can't be runtime sockets (empty stem,
                    // invalid ids) must not fail a doctor run.
                    .filter(|session| crate::local_runtime::valid_session_id(session))
                    .collect()
            })
            .unwrap_or_default(),
    };
    runtime_sessions.sort();
    // Bound the sweep: each probe costs a bounded deadline, and a directory of
    // leaked sockets must not stall `hi doctor` for minutes.
    runtime_sessions.truncate(8);
    if runtime_sessions.is_empty() {
        checks.push(Check::pass("local runtime", "no active runtime sockets"));
    }
    for runtime_session in runtime_sessions {
        let runtime_path = runtime_dir.join(format!("{runtime_session}.sock"));
        match crate::local_runtime::status_check(&runtime_session).await {
            Ok(detail) => checks.push(Check::pass("local runtime", detail)),
            Err(error) => match runtime_path.try_exists() {
                Ok(true) => checks.push(Check::fail(
                    "local runtime",
                    format!(
                        "stale or unreachable runtime socket {}: {error}",
                        runtime_path.display()
                    ),
                    "restart the local runtime leader",
                )),
                Ok(false) => checks.push(Check::pass(
                    "local runtime",
                    format!("no active runtime for session {runtime_session}"),
                )),
                Err(metadata_error) => checks.push(Check::fail(
                    "local runtime",
                    format!(
                        "cannot inspect runtime socket {}: {metadata_error}",
                        runtime_path.display()
                    ),
                    "check permissions on the runtime directory",
                )),
            },
        }
    }
    match SyncStore::status_if_available(session_id) {
        Ok(Some(status)) => checks.extend(sync_checks(&status)),
        Ok(None) => checks.push(Check::pass(
            "sync health",
            "local sync store not initialized; no local queue or lease data",
        )),
        Err(error) => checks.push(Check::fail(
            "sync health",
            format!("local sync state unavailable: {error}"),
            "check permissions and integrity of the local hi data directory",
        )),
    }
    checks
}

fn sync_checks(status: &SyncStatus) -> Vec<Check> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    let queue_age = status
        .oldest_item_unix
        .map(|oldest| now.saturating_sub(oldest));
    let topology = match (&status.lease_owner, status.lease_expiry_unix > now) {
        (Some(owner), true) => format!(
            "process mode={} · daemon/host lease owner={} generation={} expires_in={}s",
            status.mode.as_str(),
            owner,
            status.lease_generation,
            status.lease_expiry_unix.saturating_sub(now)
        ),
        _ => format!(
            "process mode={} · local process (no active host lease)",
            status.mode.as_str()
        ),
    };
    let queue = format!(
        "rows={} · bytes={} · oldest_age={} · next_retry={}",
        status.queue_rows,
        status.queue_bytes,
        queue_age
            .map(|age| format!("{age}s"))
            .unwrap_or_else(|| "none".into()),
        status
            .next_retry_unix
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".into())
    );
    let sync = format!(
        "cursor={} · quarantined={} · event_drops={} · last_success={} · last_error={}",
        status.server_cursor,
        status.quarantined_records,
        status.event_drops,
        status
            .last_success_unix
            .map(|value| value.to_string())
            .unwrap_or_else(|| "never".into()),
        status.last_error.as_deref().unwrap_or("none")
    );
    vec![
        Check::pass("runtime topology", topology),
        Check::pass("sync queue", queue),
        if status.quarantined_records > 0 || status.last_error.is_some() {
            Check::fail(
                "sync health",
                sync,
                "run `/sync status`; retry transient failures or purge quarantined records after review",
            )
        } else {
            Check::pass("sync health", sync)
        },
    ]
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

Checks config discovery, credentials, git/workspace, local runtime topology,
sync health, and (when configured) the MCP endpoint. Exits 1 when any check fails.
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
        assert!(report.checks.iter().any(|c| c.label == "client version"));
        assert!(report.checks.iter().any(|c| c.label == "sync health"));
    }

    #[test]
    fn sync_checks_report_topology_queue_cursor_and_errors() {
        let status = SyncStatus {
            mode: crate::sync_store::SyncMode::On,
            queue_rows: 3,
            queue_bytes: 42,
            oldest_item_unix: Some(1),
            last_success_unix: Some(2),
            last_error: Some("lease lost".into()),
            next_retry_unix: Some(3),
            quarantined_records: 1,
            server_cursor: 7,
            lease_generation: 4,
            lease_owner: Some("host-a".into()),
            lease_expiry_unix: u64::MAX,
            event_drops: 2,
        };

        let checks = sync_checks(&status);
        let topology = checks
            .iter()
            .find(|check| check.label == "runtime topology")
            .unwrap();
        assert!(topology.detail.as_deref().unwrap().contains("host-a"));
        let queue = checks
            .iter()
            .find(|check| check.label == "sync queue")
            .unwrap();
        assert!(queue.detail.as_deref().unwrap().contains("rows=3"));
        let health = checks
            .iter()
            .find(|check| check.label == "sync health")
            .unwrap();
        assert!(!health.passed);
        let detail = health.detail.as_deref().unwrap();
        assert!(detail.contains("cursor=7"));
        assert!(detail.contains("quarantined=1"));
        assert!(detail.contains("lease lost"));
    }

    #[test]
    fn runtime_checks_remain_structured_in_json() {
        let report = DoctorReport::from_checks(vec![Check::pass("client version", "hi 1.2.3")]);
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["checks"][0]["label"], "client version");
        assert_eq!(json["checks"][0]["detail"], "hi 1.2.3");
    }
}
