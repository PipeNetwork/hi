//! Runtime health checks for common setup failures (`hi doctor` / `/doctor`).
//!
//! Structured pass/fail checks with actionable hints, shared by the CLI one-shot
//! and in-session slash command. Frontends supply resolved settings/context and
//! render the report (human text or JSON).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::Serialize;

// ── Report types ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Check {
    pub label: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl Check {
    pub fn pass(label: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            passed: true,
            detail: Some(detail.into()),
            hint: None,
        }
    }

    pub fn fail(
        label: impl Into<String>,
        detail: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            passed: false,
            detail: Some(detail.into()),
            hint: Some(hint.into()),
        }
    }

    pub fn fail_no_hint(label: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            passed: false,
            detail: Some(detail.into()),
            hint: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DoctorReport {
    pub checks: Vec<Check>,
    pub healthy_count: usize,
    pub failing_count: usize,
}

impl DoctorReport {
    pub fn from_checks(checks: Vec<Check>) -> Self {
        let healthy_count = checks.iter().filter(|c| c.passed).count();
        let failing_count = checks.len() - healthy_count;
        Self {
            checks,
            healthy_count,
            failing_count,
        }
    }
}

/// Inputs the frontend already knows; doctor never loads config itself.
#[derive(Debug, Clone, Default)]
pub struct DoctorInput {
    pub cwd: PathBuf,
    pub workspace_root: Option<PathBuf>,
    /// Display path for a project-local config file, if present.
    pub project_config: Option<PathBuf>,
    /// Display path for the user/global config file, if present.
    pub user_config: Option<PathBuf>,
    pub provider_label: Option<String>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    /// Masked credential summary, e.g. `api_key abcd…wxyz`. `None` = missing.
    pub credentials: Option<String>,
    pub credentials_ok: bool,
    pub verify_summary: Option<String>,
    pub lsp_summary: Option<String>,
    pub checkpoint_count: Option<usize>,
    /// Optional MCP probe result already performed by the frontend.
    pub mcp: Option<Check>,
    /// Settings/config resolution failure (cold start).
    pub settings_error: Option<String>,
}

// ── Run ─────────────────────────────────────────────────────────

/// Build a doctor report from frontend-supplied facts (no I/O beyond git/fs probes).
pub fn run_doctor(input: &DoctorInput) -> DoctorReport {
    let mut checks = Vec::new();

    match &input.project_config {
        Some(path) => checks.push(Check::pass("project config", path.display().to_string())),
        None => checks.push(Check::pass("project config", "no ./hi.toml (optional)")),
    }
    match &input.user_config {
        Some(path) => checks.push(Check::pass("user config", path.display().to_string())),
        None => checks.push(Check::pass(
            "user config",
            "no user config.toml (optional until first setup)",
        )),
    }

    checks.push(check_git(&input.cwd));
    checks.push(check_workspace(
        input.workspace_root.as_deref().unwrap_or(&input.cwd),
    ));

    if let Some(err) = &input.settings_error {
        checks.push(Check::fail(
            "settings resolve",
            err.clone(),
            "run `hi` with no args on a TTY for interactive setup, or set provider/model/api_key",
        ));
    } else {
        if let Some(provider) = &input.provider_label {
            let model = input.model.as_deref().unwrap_or("(unset)");
            checks.push(Check::pass(
                "provider",
                format!("{provider} · model {model}"),
            ));
        }
        match &input.base_url {
            Some(url) if !url.trim().is_empty() => {
                checks.push(Check::pass("base_url", url.clone()));
            }
            Some(_) | None if input.provider_label.is_some() => {
                checks.push(Check::fail(
                    "base_url",
                    "empty",
                    "set base_url in the active profile or pass --base-url",
                ));
            }
            _ => {}
        }
        // Only judge credentials when the frontend resolved a provider route.
        if input.provider_label.is_some() {
            if input.credentials_ok {
                checks.push(Check::pass(
                    "credentials",
                    input
                        .credentials
                        .clone()
                        .unwrap_or_else(|| "present".into()),
                ));
            } else {
                checks.push(Check::fail(
                    "credentials",
                    input
                        .credentials
                        .clone()
                        .unwrap_or_else(|| "no API key resolved".into()),
                    "run `hi` setup, pass --api-key, or set the provider env var / auth store",
                ));
            }
        }
    }

    if let Some(verify) = &input.verify_summary {
        checks.push(Check::pass("verify command", verify.clone()));
    }
    if let Some(lsp) = &input.lsp_summary {
        let lower = lsp.to_ascii_lowercase();
        if lower.contains("error") || lower.contains("failed") {
            checks.push(Check::fail_no_hint("lsp", lsp.clone()));
        } else {
            checks.push(Check::pass("lsp", lsp.clone()));
        }
    }
    if let Some(n) = input.checkpoint_count {
        checks.push(Check::pass(
            "checkpoints",
            format!("{n} recorded this session"),
        ));
    }
    if let Some(mcp) = &input.mcp {
        checks.push(mcp.clone());
    }

    DoctorReport::from_checks(checks)
}

pub fn render_report_text(report: &DoctorReport) -> String {
    let mut out = String::from("hi doctor\n\n");
    for check in &report.checks {
        let mark = if check.passed { "ok" } else { "FAIL" };
        out.push_str(&format!("  [{mark}] {}", check.label));
        if let Some(detail) = &check.detail {
            if !detail.is_empty() {
                out.push_str(&format!(" — {detail}"));
            }
        }
        out.push('\n');
        if let Some(hint) = &check.hint {
            out.push_str(&format!("         → {hint}\n"));
        }
    }
    out.push('\n');
    out.push_str(&format!(
        "Found {} healthy, {} failing.",
        report.healthy_count, report.failing_count
    ));
    if report.failing_count > 0 {
        out.push_str(" Re-run with `hi doctor --json` for machine-readable output.");
    }
    out.push('\n');
    out
}

pub fn print_report(report: &DoctorReport) {
    print!("{}", render_report_text(report));
}

// ── Local probes ────────────────────────────────────────────────

fn check_git(cwd: &Path) -> Check {
    let version = Command::new("git")
        .arg("--version")
        .current_dir(cwd)
        .output();
    match version {
        Ok(out) if out.status.success() => {
            let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let inside = Command::new("git")
                .args(["rev-parse", "--is-inside-work-tree"])
                .current_dir(cwd)
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim() == "true")
                .unwrap_or(false);
            if inside {
                Check::pass("git", format!("{ver}; inside work tree"))
            } else {
                Check::pass("git", format!("{ver}; not a git work tree (optional)"))
            }
        }
        Ok(out) => Check::fail(
            "git",
            format!(
                "git --version exited {}",
                out.status.code().unwrap_or_default()
            ),
            "install git and ensure it is on PATH",
        ),
        Err(err) => Check::fail(
            "git",
            err.to_string(),
            "install git and ensure it is on PATH",
        ),
    }
}

fn check_workspace(root: &Path) -> Check {
    let meta = match std::fs::metadata(root) {
        Ok(m) => m,
        Err(err) => {
            return Check::fail(
                "workspace",
                format!("{}: {err}", root.display()),
                "cd into a readable project directory",
            );
        }
    };
    if !meta.is_dir() {
        return Check::fail(
            "workspace",
            format!("{} is not a directory", root.display()),
            "cd into a project directory",
        );
    }
    let probe_dir = root.join(".hi");
    let probe_ok = if probe_dir.is_dir() || std::fs::create_dir_all(&probe_dir).is_ok() {
        let probe = probe_dir.join(".doctor-write-probe");
        let wrote = std::fs::write(&probe, b"ok").is_ok();
        let _ = std::fs::remove_file(&probe);
        wrote
    } else {
        false
    };
    if probe_ok {
        Check::pass("workspace", format!("{} (writable)", root.display()))
    } else {
        Check::fail(
            "workspace",
            format!("{} not writable", root.display()),
            "check directory permissions",
        )
    }
}

/// Probe a Pipe MCP endpoint; used by CLI/TUI frontends before assembling input.
pub async fn probe_mcp(url: &str, api_key: &str, current_model: &str) -> Check {
    let fut = async {
        let client = hi_ai::PipeMcpClient::new(url, api_key.to_string());
        let (server, _protocol) = client.server_info().await?;
        let tools = client.tools_list().await?;
        let _models = client.list_models().await?;
        let _ = current_model;
        anyhow::Ok((server, tools.len()))
    };
    match tokio::time::timeout(Duration::from_secs(12), fut).await {
        Ok(Ok((server, tools))) if tools > 0 => {
            Check::pass("mcp endpoint", format!("{server}; {tools} tools · {url}"))
        }
        Ok(Ok((server, _))) => Check::fail(
            "mcp endpoint",
            format!("{server}; 0 tools discovered"),
            "check MCP server config and credentials",
        ),
        Ok(Err(err)) => Check::fail(
            "mcp endpoint",
            err.to_string(),
            "verify mcp_url / API key, or unset mcp_url if unused",
        ),
        Err(_) => Check::fail(
            "mcp endpoint",
            format!("timed out after 12s contacting {url}"),
            "check network connectivity and mcp_url",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_counts_failures() {
        let report = DoctorReport::from_checks(vec![
            Check::pass("a", "ok"),
            Check::fail("b", "nope", "fix it"),
        ]);
        assert_eq!(report.healthy_count, 1);
        assert_eq!(report.failing_count, 1);
    }

    #[test]
    fn render_includes_hints() {
        let report = DoctorReport::from_checks(vec![Check::fail(
            "credentials",
            "missing",
            "set an API key",
        )]);
        let text = render_report_text(&report);
        assert!(text.contains("[FAIL] credentials"));
        assert!(text.contains("set an API key"));
        assert!(text.contains("Found 0 healthy, 1 failing"));
    }

    #[test]
    fn run_doctor_surfaces_settings_error() {
        let report = run_doctor(&DoctorInput {
            cwd: PathBuf::from("."),
            settings_error: Some("no model configured".into()),
            ..DoctorInput::default()
        });
        assert!(report.failing_count >= 1);
        assert!(
            report
                .checks
                .iter()
                .any(|c| c.label == "settings resolve" && !c.passed)
        );
    }

    #[test]
    fn run_doctor_ok_path_includes_provider() {
        let report = run_doctor(&DoctorInput {
            cwd: PathBuf::from("."),
            provider_label: Some("xai".into()),
            model: Some("grok".into()),
            base_url: Some("https://api.x.ai/v1".into()),
            credentials: Some("api_key abcd…wxyz".into()),
            credentials_ok: true,
            ..DoctorInput::default()
        });
        assert!(
            report
                .checks
                .iter()
                .any(|c| c.label == "provider" && c.passed)
        );
        assert!(
            report
                .checks
                .iter()
                .any(|c| c.label == "credentials" && c.passed)
        );
    }
}
