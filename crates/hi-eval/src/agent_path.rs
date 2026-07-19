//! Agent-level eval path: full `hi` binary + verify-in-the-loop, not bare `hi-ai`.
//!
//! The main matrix (`run_config`) already shells out to `hi` with `--verify` /
//! `--report`. This module makes that contract **explicit and testable**:
//! harness flags, report verification shape, and the baseline↔verify A/B.
//!
//! Use `--agent-path` for a model-free smoke of the wiring (fake `hi` script),
//! or the normal matrix with `--configs=verify` against a real model.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};

/// Report keys the agent-level path must surface so eval can distinguish
/// "model stopped" from "repair verifier ran".
pub const REQUIRED_REPORT_KEYS: &[&str] = &["schema_version", "outcome", "verification", "tools"];

/// Sub-keys under `verification` written by `hi --report` after a turn.
pub const REQUIRED_VERIFICATION_KEYS: &[&str] = &["mode", "status", "stages", "rounds"];

/// Validate a `hi --report` JSON document carries agent-level verification
/// evidence (not merely a completion blob).
pub fn validate_agent_report(value: &serde_json::Value) -> Result<()> {
    ensure!(
        value.get("schema_version").and_then(|v| v.as_u64()) == Some(2),
        "agent report requires schema_version 2"
    );
    for key in REQUIRED_REPORT_KEYS {
        ensure!(
            value.get(*key).is_some(),
            "agent report missing top-level key `{key}`"
        );
    }
    let verification = value
        .get("verification")
        .context("agent report missing verification block")?;
    for key in REQUIRED_VERIFICATION_KEYS {
        ensure!(
            verification.get(*key).is_some(),
            "agent report verification missing `{key}`"
        );
    }
    let outcome = value
        .get("outcome")
        .context("agent report missing outcome")?;
    ensure!(
        outcome.get("verification").is_some(),
        "agent report outcome missing verification status"
    );
    ensure!(
        outcome.get("status").is_some(),
        "agent report outcome missing status"
    );
    Ok(())
}

/// Whether this report shows the repair verifier actually executed at least
/// one stage (as opposed to skipped/off).
pub fn report_ran_verification_stages(value: &serde_json::Value) -> bool {
    value
        .get("verification")
        .and_then(|v| v.get("stages"))
        .and_then(|s| s.as_array())
        .is_some_and(|stages| !stages.is_empty())
}

/// Model-free smoke: a fake `hi` that records argv and writes a minimal v2
/// report when `--report` is present. Asserts the harness passes `--verify`
/// and that [`validate_agent_report`] accepts the artifact shape.
pub fn run_agent_path_smoke() -> Result<()> {
    let dir = tempfile_dir()?;
    let hi = write_fake_hi(&dir)?;
    let report = dir.join("report.json");
    let work = dir.join("work");
    std::fs::create_dir_all(&work)?;

    let status = Command::new(&hi)
        .current_dir(&work)
        .arg("--report")
        .arg(&report)
        .arg("--verify")
        .arg("true")
        .arg("--no-save")
        .arg("agent-path smoke prompt")
        .status()
        .context("launching fake hi for agent-path smoke")?;
    ensure!(status.success(), "fake hi exited non-zero: {status}");

    let raw = std::fs::read_to_string(&report).context("reading fake hi report")?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).context("parsing fake hi report JSON")?;
    validate_agent_report(&value)?;
    ensure!(
        report_ran_verification_stages(&value),
        "fake hi report should include verification stages"
    );

    let argv = std::fs::read_to_string(dir.join("argv.txt")).context("reading fake hi argv")?;
    ensure!(
        argv.contains("--verify"),
        "agent-path smoke expected --verify on argv, got:\n{argv}"
    );
    ensure!(
        argv.contains("--report"),
        "agent-path smoke expected --report on argv, got:\n{argv}"
    );
    eprintln!(
        "hi-eval --agent-path: ok (report schema + --verify wiring via fake hi at {})",
        hi.display()
    );
    Ok(())
}

fn tempfile_dir() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!(
        "hi-eval-agent-path-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn write_fake_hi(dir: &Path) -> Result<PathBuf> {
    let path = dir.join("fake-hi");
    let report_body = r#"{"schema_version":2,"outcome":{"status":"completed","verification":"passed","review":"not_required","stop_reason":"completed","changed_files":[],"effective_route":{"provider":"fake","model":"test"}},"verification":{"mode":"explicit","status":"passed","planned_stages":[{"name":"check","command":"true"}],"stages":[{"round":1,"name":"check","command":"true","status":"succeeded"}],"rounds":1,"attributions":[]},"tools":[],"usage":{"turn":{"input_tokens":1,"output_tokens":1,"total_tokens":2}},"telemetry":{}}"#;
    let body_path = dir.join("report_body.json");
    std::fs::write(&body_path, report_body)?;
    let script = format!(
        r#"#!/bin/sh
set -e
printf '%s\n' "$@" > "{argv}"
report=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--report" ]; then report="$arg"; fi
  prev="$arg"
done
if [ -n "$report" ]; then
  cp "{body}" "$report"
fi
"#,
        argv = dir.join("argv.txt").display(),
        body = body_path.display(),
    );
    std::fs::write(&path, script)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms)?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_minimal_v2_agent_report() {
        let value = serde_json::json!({
            "schema_version": 2,
            "outcome": {
                "status": "completed",
                "verification": "passed"
            },
            "verification": {
                "mode": "auto",
                "status": "passed",
                "stages": [{"round": 1, "name": "test", "command": "true", "status": "succeeded"}],
                "rounds": 1
            },
            "tools": []
        });
        validate_agent_report(&value).unwrap();
        assert!(report_ran_verification_stages(&value));
    }

    #[test]
    fn rejects_completion_only_blob() {
        let value = serde_json::json!({
            "text": "done",
            "tokens": 3
        });
        assert!(validate_agent_report(&value).is_err());
    }

    #[test]
    fn agent_path_smoke_runs() {
        run_agent_path_smoke().expect("agent-path smoke");
    }
}
