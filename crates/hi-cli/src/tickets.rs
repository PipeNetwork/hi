//! `hi tickets` — claim project tickets and run `hi --goal --verify` until the
//! report passes or the ticket budget is gone.
//!
//! This is the opposite direction of `POST /v1/tasks`: the control plane holds
//! the work item; this daemon leases a local ticket and executes it in the
//! current working directory. Session `--daemon` free-text input is not a
//! ticket (no lease, no contract).

use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Child;

const HEARTBEAT_SECS: u64 = 30;
const IDLE_POLL_SECS: u64 = 5;

pub async fn run_cli(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }
    if !args.is_empty() {
        bail!("hi tickets takes no arguments — run it in the repo after `/login pipenetwork`");
    }
    run_daemon().await
}

fn print_help() {
    println!(
        "\
hi tickets — claim Pipe project tickets and run them locally

Pair a project key (`hi /login pipenetwork`, pick the project in the
dashboard), cd into the repo, then:

  hi tickets

The daemon heartbeats, leases one queued local ticket at a time, and
spawns `hi --goal … --verify …` in this directory. Sandbox tickets are
not executed here.

Environment:
  PIPENETWORK_API_KEY   project key (or auth.json after pairing)
  PIPENETWORK_API_BASE  control plane origin (default https://api.pipenetwork.ai)
"
    );
}

async fn run_daemon() -> Result<()> {
    let creds = TicketCredentials::resolve()?;
    let identity = AgentIdentity::capture()?;
    let client = TicketClient::new(creds)?;
    let agent = client.heartbeat(&identity).await?;
    println!(
        "hi tickets (pid {}) — agent {} on {} in {}",
        std::process::id(),
        agent.id,
        identity.hostname,
        identity.cwd
    );
    println!("Claiming local tickets for this project. Ctrl-C to stop.");

    loop {
        tokio::select! {
            _ = shutdown_signal() => {
                println!("hi tickets stopping");
                return Ok(());
            }
            result = run_idle_cycle(&client, &identity) => {
                result?;
            }
        }
    }
}

async fn run_idle_cycle(client: &TicketClient, identity: &AgentIdentity) -> Result<()> {
    let agent = client.heartbeat(identity).await?;
    let Some(ticket) = client.claim(identity).await? else {
        tokio::time::sleep(Duration::from_secs(IDLE_POLL_SECS)).await;
        return Ok(());
    };
    println!("claimed {} — {}", ticket.id, ticket.title);
    execute_ticket(client, identity, &agent.id, &ticket).await
}

async fn execute_ticket(
    client: &TicketClient,
    identity: &AgentIdentity,
    agent_id: &str,
    ticket: &TicketView,
) -> Result<()> {
    let session = crate::session::new_fleet_session_path()?;
    let report = session.with_extension("report.json");
    let _ = std::fs::remove_file(&report);
    let mut child = spawn_goal_child(ticket, &session, &report)?;
    let pid = child.id();
    let _ = client
        .progress(
            agent_id,
            &ticket.id,
            "turn.started",
            Some(json!({ "session": session.display().to_string(), "pid": pid })),
        )
        .await;
    loop {
        tokio::select! {
            _ = shutdown_signal() => {
                let _ = child.kill().await;
                let remaining = ticket_remaining_usd(ticket);
                let status = if remaining < 0.01 {
                    "budget_exhausted"
                } else {
                    "repairing"
                };
                client
                    .complete(agent_id, &ticket.id, status, None, None, Some("daemon interrupted"))
                    .await?;
                bail!("interrupted while running {}", ticket.id);
            }
            _ = tokio::time::sleep(Duration::from_secs(HEARTBEAT_SECS)) => {
                let _ = client.heartbeat(identity).await;
            }
            wait = child.wait() => {
                let ok = wait.ok().and_then(|status| status.code()).unwrap_or(1) == 0;
                finish_ticket(client, agent_id, ticket, &report, ok).await?;
                return Ok(());
            }
        }
    }
}

async fn finish_ticket(
    client: &TicketClient,
    agent_id: &str,
    ticket: &TicketView,
    report_path: &Path,
    child_ok: bool,
) -> Result<()> {
    let report = std::fs::read_to_string(report_path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());
    let (status, summary) = if let Some(report) = report.as_ref() {
        let status = status_from_hi_report(report);
        let summary = report
            .pointer("/outcome/stop_reason")
            .and_then(Value::as_str)
            .map(str::to_owned);
        (status, summary)
    } else if child_ok {
        (
            "failed".into(),
            Some("child exited without a report".into()),
        )
    } else if ticket.attempt_count >= ticket.maximum_attempts.max(1)
        || ticket_remaining_usd(ticket) < 0.01
    {
        (
            if ticket_remaining_usd(ticket) < 0.01 {
                "budget_exhausted"
            } else {
                "failed"
            }
            .into(),
            Some("child crashed with no remaining budget or attempts".into()),
        )
    } else {
        (
            "repairing".into(),
            Some("child crashed; retrying with remaining budget".into()),
        )
    };
    if status == "failed" {
        let _ = client
            .progress(agent_id, &ticket.id, "verification.failed", report.clone())
            .await;
    }
    client
        .complete(
            agent_id,
            &ticket.id,
            &status,
            report,
            None,
            summary.as_deref(),
        )
        .await?;
    println!("{} → {status}", ticket.id);
    Ok(())
}

fn spawn_goal_child(ticket: &TicketView, session: &Path, report: &Path) -> Result<Child> {
    let exe = std::env::current_exe().context("could not locate the hi binary")?;
    let argv = child_argv(ticket, session, report);
    let mut cmd = tokio::process::Command::new(exe);
    cmd.args(&argv)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if ticket.acceptance.qa == "skeptic" {
        cmd.env("HI_GOAL_TEAM", "1");
    }
    if let Ok(key) = std::env::var("PIPENETWORK_API_KEY")
        && !key.trim().is_empty()
    {
        cmd.env("PIPENETWORK_API_KEY", key);
    }
    let mut child = cmd.spawn().context("failed to spawn hi --goal")?;
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                println!("{line}");
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("{line}");
            }
        });
    }
    Ok(child)
}

fn child_argv(ticket: &TicketView, session: &Path, report: &Path) -> Vec<String> {
    let mut argv = vec![
        "--provider".into(),
        "pipenetwork".into(),
        "--session-file".into(),
        session.display().to_string(),
        "--report".into(),
        report.display().to_string(),
        "--goal".into(),
        ticket.goal.clone(),
    ];
    if let Some(verify) = ticket
        .acceptance
        .verify_commands
        .iter()
        .map(|command| command.trim())
        .find(|command| !command.is_empty())
    {
        argv.push("--verify".into());
        argv.push(verify.to_string());
    }
    argv
}

fn ticket_remaining_usd(ticket: &TicketView) -> f64 {
    (ticket.maximum_cost_usd - ticket.spent_usd).max(0.0)
}

/// Mirror of ipop-core `ticket_status_from_hi_report` (string form for complete).
fn status_from_hi_report(report: &Value) -> String {
    let verification = report
        .pointer("/outcome/verification")
        .or_else(|| report.pointer("/verification/status"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let status = report
        .pointer("/outcome/status")
        .and_then(Value::as_str)
        .unwrap_or("");
    let stop_reason = report
        .pointer("/outcome/stop_reason")
        .and_then(Value::as_str)
        .unwrap_or("");
    let leftover = report
        .pointer("/goal/drive")
        .and_then(Value::as_str)
        .unwrap_or("");
    let stopped_at_work_limit = matches!(stop_reason, "step_limit" | "tool_limit");
    if status == "cancelled" {
        return "canceled".into();
    }
    // Failure evidence outranks a coincident/legacy limit reason. Older
    // reports can contain `failed + step_limit`; a failed check must not be
    // presented as resumable repair work.
    if matches!(verification, "failed" | "infrastructure_error") {
        return "failed".into();
    }
    // A finite model- or tool-call ceiling means the child stopped with work
    // still pending even when verification happened to pass for the partial
    // workspace. Do not close the ticket as successful merely because the
    // generic outcome classifier reports `completed` after clean settlement.
    if stopped_at_work_limit && matches!(verification, "passed" | "not_applicable") {
        return "repairing".into();
    }
    if verification == "passed" && matches!(status, "completed" | "") && leftover != "active" {
        return "succeeded".into();
    }
    if leftover == "active"
        // Schema-v2 reports written before the outcome migration remain valid
        // input, but new reports never emit this value.
        || status == "incomplete"
    {
        return "repairing".into();
    }
    "failed".into()
}

struct TicketCredentials {
    origin: String,
    api_key: String,
}

impl TicketCredentials {
    fn resolve() -> Result<Self> {
        let origin = hi_ai::pipenetwork_auth::api_base();
        let api_key = std::env::var("PIPENETWORK_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                std::env::var("HI_API_KEY")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
            .or_else(|| {
                hi_ai::auth_store::load(hi_ai::pipenetwork_auth::PROVIDER_ID)
                    .map(|stored| stored.access)
                    .filter(|value| !value.trim().is_empty())
            })
            .ok_or_else(|| {
                anyhow!(
                    "no Pipe project key — run `hi /login pipenetwork` or set PIPENETWORK_API_KEY"
                )
            })?;
        Ok(Self { origin, api_key })
    }
}

struct AgentIdentity {
    machine_id: String,
    hostname: String,
    cwd: String,
}

impl AgentIdentity {
    fn capture() -> Result<Self> {
        let cwd = std::env::current_dir()
            .context("could not read cwd")?
            .display()
            .to_string();
        let hostname = hostname();
        let machine_id = std::env::var("HI_TICKET_MACHINE_ID")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(machine_id_from_host)
            .unwrap_or_else(|| hostname.clone());
        Ok(Self {
            machine_id,
            hostname,
            cwd,
        })
    }

    fn body(&self) -> Value {
        json!({
            "machine_id": self.machine_id,
            "hostname": self.hostname,
            "cwd": self.cwd,
        })
    }
}

pub(crate) fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::process::Command::new("hostname")
                .output()
                .ok()
                .and_then(|output| {
                    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    (!text.is_empty()).then_some(text)
                })
        })
        .unwrap_or_else(|| "unknown".into())
}

fn machine_id_from_host() -> Option<String> {
    std::fs::read_to_string("/etc/machine-id")
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

struct TicketClient {
    http: reqwest::Client,
    origin: String,
    api_key: String,
}

impl TicketClient {
    fn new(creds: TicketCredentials) -> Result<Self> {
        let http = reqwest::Client::builder()
            .redirect(hi_ai::credential_redirect_policy())
            .http1_only()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(5, 30));
        Ok(Self {
            http,
            origin: creds.origin,
            api_key: creds.api_key,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.origin.trim_end_matches('/'))
    }

    async fn send_json<T: for<'de> Deserialize<'de>>(
        &self,
        builder: reqwest::RequestBuilder,
        extra: Option<(&str, &str)>,
    ) -> Result<T> {
        let mut builder = builder.bearer_auth(&self.api_key);
        if let Some((name, value)) = extra {
            builder = builder.header(name, value);
        }
        let response = builder.send().await.context("ticket API request failed")?;
        let status = response.status();
        let bytes = response.bytes().await.context("ticket API body")?;
        if !status.is_success() {
            bail!("ticket API {status}: {}", String::from_utf8_lossy(&bytes));
        }
        serde_json::from_slice(&bytes).context("decoding ticket API response")
    }

    async fn heartbeat(&self, identity: &AgentIdentity) -> Result<AgentView> {
        self.send_json(
            self.http
                .post(self.url("/v1/hi/agents/heartbeat"))
                .json(&identity.body()),
            None,
        )
        .await
    }

    async fn claim(&self, identity: &AgentIdentity) -> Result<Option<TicketView>> {
        let response: ClaimResponse = self
            .send_json(
                self.http
                    .post(self.url("/v1/hi/tickets/claim"))
                    .json(&identity.body()),
                None,
            )
            .await?;
        Ok(response.ticket)
    }

    async fn progress(
        &self,
        agent_id: &str,
        ticket_id: &str,
        event_type: &str,
        payload: Option<Value>,
    ) -> Result<()> {
        let _: Value = self
            .send_json(
                self.http
                    .post(self.url(&format!("/v1/hi/tickets/{ticket_id}/events")))
                    .json(&json!({ "type": event_type, "payload": payload })),
                Some(("x-hi-agent-id", agent_id)),
            )
            .await?;
        Ok(())
    }

    async fn complete(
        &self,
        agent_id: &str,
        ticket_id: &str,
        status: &str,
        report: Option<Value>,
        spent_usd: Option<f64>,
        summary: Option<&str>,
    ) -> Result<()> {
        let _: Value = self
            .send_json(
                self.http
                    .post(self.url(&format!("/v1/hi/tickets/{ticket_id}/complete")))
                    .json(&json!({
                        "status": status,
                        "report": report,
                        "spent_usd": spent_usd,
                        "summary": summary,
                    })),
                Some(("x-hi-agent-id", agent_id)),
            )
            .await?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct AgentView {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ClaimResponse {
    #[serde(default)]
    ticket: Option<TicketView>,
}

#[derive(Debug, Clone, Deserialize)]
struct TicketView {
    id: String,
    title: String,
    goal: String,
    #[serde(default)]
    acceptance: TicketAcceptance,
    #[serde(default)]
    maximum_cost_usd: f64,
    #[serde(default)]
    spent_usd: f64,
    #[serde(default)]
    attempt_count: u32,
    #[serde(default = "default_maximum_attempts")]
    maximum_attempts: u32,
}

fn default_maximum_attempts() -> u32 {
    3
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TicketAcceptance {
    #[serde(default)]
    verify_commands: Vec<String>,
    #[serde(default)]
    qa: String,
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ticket(verify: &str, qa: &str) -> TicketView {
        TicketView {
            id: "ticket_1".into(),
            title: "fix parser".into(),
            goal: "make tests pass".into(),
            acceptance: TicketAcceptance {
                verify_commands: vec![verify.into()],
                qa: qa.into(),
            },
            maximum_cost_usd: 5.0,
            spent_usd: 0.0,
            attempt_count: 1,
            maximum_attempts: 3,
        }
    }

    #[test]
    fn child_argv_includes_goal_verify_and_report() {
        let session = PathBuf::from("/tmp/s.jsonl");
        let report = PathBuf::from("/tmp/s.report.json");
        let argv = child_argv(&sample_ticket("cargo test", "off"), &session, &report);
        assert!(
            argv.windows(2)
                .any(|pair| pair == ["--goal", "make tests pass"])
        );
        assert!(
            argv.windows(2)
                .any(|pair| pair == ["--verify", "cargo test"])
        );
        assert!(
            !argv.iter().any(|arg| arg == "--max-verify-repairs"),
            "ticket children inherit the ordinary unlimited repair default"
        );
        assert!(argv.contains(&"--session-file".into()));
        assert!(argv.contains(&"--report".into()));
    }

    #[test]
    fn child_argv_omits_verify_when_empty() {
        let ticket = sample_ticket("", "skeptic");
        let argv = child_argv(
            &ticket,
            Path::new("/tmp/s.jsonl"),
            Path::new("/tmp/s.report.json"),
        );
        assert!(!argv.contains(&"--verify".into()));
    }

    #[test]
    fn report_passed_maps_to_succeeded() {
        let report = json!({
            "outcome": { "status": "completed", "verification": "passed" }
        });
        assert_eq!(status_from_hi_report(&report), "succeeded");
    }

    #[test]
    fn report_failed_verify_maps_to_failed() {
        let report = json!({
            "outcome": { "status": "completed", "verification": "failed" }
        });
        assert_eq!(status_from_hi_report(&report), "failed");
    }

    #[test]
    fn leftover_goal_drive_maps_to_repairing() {
        let report = json!({
            "outcome": { "status": "completed", "verification": "passed", "stop_reason": "step_limit" },
            "goal": { "drive": "active" }
        });
        assert_eq!(status_from_hi_report(&report), "repairing");
    }

    #[test]
    fn canonicalized_legacy_limit_failure_maps_to_repairing() {
        let report = json!({
            "outcome": { "status": "failed", "verification": "passed", "stop_reason": "step_limit" }
        });
        assert_eq!(status_from_hi_report(&report), "repairing");
    }

    #[test]
    fn completed_step_limit_does_not_close_unfinished_ticket() {
        let report = json!({
            "outcome": { "status": "completed", "verification": "passed", "stop_reason": "step_limit" }
        });
        assert_eq!(status_from_hi_report(&report), "repairing");
    }

    #[test]
    fn completed_tool_limit_does_not_close_unfinished_ticket() {
        let report = json!({
            "outcome": { "status": "completed", "verification": "passed", "stop_reason": "tool_limit" }
        });
        assert_eq!(status_from_hi_report(&report), "repairing");
    }

    #[test]
    fn tool_limit_failure_with_clean_partial_workspace_is_repairing() {
        let report = json!({
            "outcome": { "status": "failed", "verification": "not_applicable", "stop_reason": "tool_limit" }
        });
        assert_eq!(status_from_hi_report(&report), "repairing");
    }

    #[test]
    fn failed_verification_outranks_a_legacy_limit_reason() {
        let report = json!({
            "outcome": { "status": "failed", "verification": "failed", "stop_reason": "step_limit" }
        });
        assert_eq!(status_from_hi_report(&report), "failed");
    }

    #[test]
    fn failed_verification_outranks_tool_limit() {
        let report = json!({
            "outcome": { "status": "failed", "verification": "failed", "stop_reason": "tool_limit" }
        });
        assert_eq!(status_from_hi_report(&report), "failed");
    }

    #[test]
    fn legacy_incomplete_report_still_maps_to_repairing() {
        let report = json!({
            "outcome": { "status": "incomplete", "verification": "passed", "stop_reason": "stalled" }
        });
        assert_eq!(status_from_hi_report(&report), "repairing");
    }
}
