//! Unsandboxed layered cargo + isolated reproduction. Never the live user workspace.

use std::path::{Path, PathBuf};
use std::process::Output;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::fsutil;
use crate::spawn;

const LAYER0: Duration = Duration::from_secs(5 * 60);
const LAYER12: Duration = Duration::from_secs(5 * 60);
const LAYER345: Duration = Duration::from_secs(8 * 60);
const CLIPPY: Duration = Duration::from_secs(5 * 60);

#[derive(Clone, Debug)]
pub struct GateInput {
    pub worktree: PathBuf,
    pub head_worktree: PathBuf,
    pub incident_dir: PathBuf,
    pub cargo_home: PathBuf,
    pub cargo_target_dir: PathBuf,
    pub head_target_dir: PathBuf,
    pub failing_test: Option<Vec<String>>,
    pub timeouts: GateTimeouts,
    /// Worktree commit before the repair commit. `git diff HEAD` is empty after `commit_fix`.
    pub base_sha: String,
}

#[derive(Clone, Debug)]
pub struct GateTimeouts {
    pub layer0: Duration,
    pub check: Duration,
    pub test: Duration,
    pub clippy: Duration,
}

impl Default for GateTimeouts {
    fn default() -> Self {
        Self {
            layer0: LAYER0,
            check: LAYER12,
            test: LAYER345,
            clippy: CLIPPY,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GateReport {
    pub passed: bool,
    pub skip_apply_reason: Option<String>,
    pub layers: Vec<LayerReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayerReport {
    pub layer: u8,
    pub command: String,
    pub passed: bool,
    pub skipped: bool,
    pub reason: Option<String>,
    pub duration_ms: u64,
}

pub async fn run_layers(input: &GateInput) -> Result<GateReport> {
    let mut layers = Vec::new();
    let offline =
        cargo_offline_warm(&input.worktree, &input.cargo_home, &input.cargo_target_dir).await;
    let layer0 = layer0(input, offline).await?;
    let skip_apply = layer0
        .reason
        .clone()
        .filter(|_| layer0.skipped || !layer0.passed);
    let layer0_ok = layer0.passed && !layer0.skipped;
    layers.push(layer0);
    if !layer0_ok {
        let report = GateReport {
            passed: false,
            skip_apply_reason: skip_apply.or(Some(
                "layer 0 did not reproduce HEAD-fail / repair-pass".into(),
            )),
            layers,
        };
        write_report(&input.incident_dir, &report)?;
        return Ok(report);
    }

    layers.push(
        run_named(
            1,
            &input.worktree,
            &input.cargo_home,
            &input.cargo_target_dir,
            cargo_args(
                offline,
                &[
                    "check",
                    "-p",
                    "hi",
                    "-p",
                    "hi-harness",
                    "-p",
                    "hi-liveness",
                    "-p",
                    "hi-sentinel",
                ],
            ),
            input.timeouts.check,
        )
        .await?,
    );

    layers.push(
        run_named(
            2,
            &input.worktree,
            &input.cargo_home,
            &input.cargo_target_dir,
            cargo_args(offline, &["test", "-p", "hi-liveness", "-p", "hi-sentinel"]),
            input.timeouts.test,
        )
        .await?,
    );

    let touched = touched_crates(&input.worktree, &input.base_sha)?;
    if touched.iter().any(|c| c == "hi-harness") {
        layers.push(
            run_named(
                3,
                &input.worktree,
                &input.cargo_home,
                &input.cargo_target_dir,
                cargo_args(offline, &["test", "-p", "hi-harness"]),
                input.timeouts.test,
            )
            .await?,
        );
    } else {
        layers.push(skipped(3, "hi-harness not in diff"));
    }

    if touched.iter().any(|c| c == "hi") {
        layers.push(
            run_named(
                4,
                &input.worktree,
                &input.cargo_home,
                &input.cargo_target_dir,
                cargo_args(offline, &["test", "-p", "hi", "--lib"]),
                input.timeouts.test,
            )
            .await?,
        );
    } else {
        layers.push(skipped(4, "hi-cli not in diff"));
    }

    for pkg in ["hi-tui", "hi-ai", "hi-tools"] {
        if touched.iter().any(|c| c == pkg) {
            layers.push(
                run_named(
                    4,
                    &input.worktree,
                    &input.cargo_home,
                    &input.cargo_target_dir,
                    cargo_args(offline, &["test", "-p", pkg]),
                    input.timeouts.test,
                )
                .await?,
            );
        }
    }

    let clippy_pkgs: Vec<&str> = [
        "hi-liveness",
        "hi-sentinel",
        "hi-harness",
        "hi",
        "hi-tui",
        "hi-ai",
        "hi-tools",
    ]
    .into_iter()
    .filter(|p| touched.iter().any(|c| c == p))
    .collect();
    if clippy_pkgs.is_empty() {
        layers.push(skipped(5, "no clippy crates in diff"));
    } else {
        let mut args = vec!["clippy".to_string()];
        if offline {
            args.insert(0, "--offline".into());
        }
        for p in clippy_pkgs {
            args.push("-p".into());
            args.push(p.to_string());
        }
        args.extend([
            "--all-targets".into(),
            "--".into(),
            "-D".into(),
            "warnings".into(),
        ]);
        layers.push(
            run_named(
                5,
                &input.worktree,
                &input.cargo_home,
                &input.cargo_target_dir,
                args,
                input.timeouts.clippy,
            )
            .await?,
        );
    }

    let passed = layers.iter().all(|l| l.passed || l.skipped);
    let report = GateReport {
        passed,
        skip_apply_reason: None,
        layers,
    };
    write_report(&input.incident_dir, &report)?;
    Ok(report)
}

async fn layer0(input: &GateInput, offline: bool) -> Result<LayerReport> {
    let started = Instant::now();
    let script = input.incident_dir.join("repro/reproduction.sh");
    if !script.is_file() {
        return Ok(LayerReport {
            layer: 0,
            command: "reproduction.sh".into(),
            passed: false,
            skipped: true,
            reason: Some("no isolated repro fixture; fail closed".into()),
            duration_ms: elapsed_ms(started),
        });
    }

    let head_bin = build_hi(
        &input.head_worktree,
        &input.cargo_home,
        &input.head_target_dir,
        offline,
        input.timeouts.layer0,
    )
    .await?;
    let repair_bin = build_hi(
        &input.worktree,
        &input.cargo_home,
        &input.cargo_target_dir,
        offline,
        input.timeouts.layer0,
    )
    .await?;

    let head = run_repro(
        &script,
        &head_bin,
        &input.head_worktree,
        &input.cargo_home,
        &input.head_target_dir,
        input.failing_test.as_deref(),
        input.timeouts.layer0,
    )
    .await?;
    if repro_fail_closed(&head) && input.failing_test.is_none() {
        return Ok(LayerReport {
            layer: 0,
            command: format!("{} HI_BINARY=HEAD", script.display()),
            passed: false,
            skipped: true,
            reason: Some("no isolated repro fixture; fail closed".into()),
            duration_ms: elapsed_ms(started),
        });
    }
    if head.status.success() {
        return Ok(LayerReport {
            layer: 0,
            command: format!("{} HI_BINARY=HEAD", script.display()),
            passed: false,
            skipped: true,
            reason: Some("HEAD-built binary did not fail".into()),
            duration_ms: elapsed_ms(started),
        });
    }

    let repair = run_repro(
        &script,
        &repair_bin,
        &input.worktree,
        &input.cargo_home,
        &input.cargo_target_dir,
        input.failing_test.as_deref(),
        input.timeouts.layer0,
    )
    .await?;
    if !repair.status.success() {
        return Ok(LayerReport {
            layer: 0,
            command: format!("{} HI_BINARY=repair", script.display()),
            passed: false,
            skipped: false,
            reason: Some("repair-built binary still fails".into()),
            duration_ms: elapsed_ms(started),
        });
    }

    if let Some(args) = &input.failing_test {
        let mut head_args = cargo_args(offline, &["test"]);
        head_args.extend(args.clone());
        let head_test = cargo_capture(
            &input.head_worktree,
            &input.cargo_home,
            &input.head_target_dir,
            &head_args,
            input.timeouts.layer0,
        )
        .await?;
        if head_test.status.success() {
            return Ok(LayerReport {
                layer: 0,
                command: format!("cargo test {}", args.join(" ")),
                passed: false,
                skipped: true,
                reason: Some("HEAD cargo test did not fail".into()),
                duration_ms: elapsed_ms(started),
            });
        }
        let mut repair_args = cargo_args(offline, &["test"]);
        repair_args.extend(args.clone());
        let repair_test = cargo_capture(
            &input.worktree,
            &input.cargo_home,
            &input.cargo_target_dir,
            &repair_args,
            input.timeouts.layer0,
        )
        .await?;
        if !repair_test.status.success() {
            return Ok(LayerReport {
                layer: 0,
                command: format!("cargo test {}", args.join(" ")),
                passed: false,
                skipped: false,
                reason: Some("repair cargo test failed".into()),
                duration_ms: elapsed_ms(started),
            });
        }
    }

    Ok(LayerReport {
        layer: 0,
        command: format!(
            "{} HI_BINARY=<worktree hi> HEAD-fail/repair-pass",
            script.display()
        ),
        passed: true,
        skipped: false,
        reason: None,
        duration_ms: elapsed_ms(started),
    })
}

async fn build_hi(
    worktree: &Path,
    cargo_home: &Path,
    target_dir: &Path,
    offline: bool,
    timeout: Duration,
) -> Result<PathBuf> {
    let args = cargo_args(offline, &["build", "-p", "hi", "-p", "hi-sentinel"]);
    let output = cargo_capture(worktree, cargo_home, target_dir, &args, timeout).await?;
    anyhow::ensure!(
        output.status.success(),
        "cargo build -p hi failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bin = target_dir.join("debug/hi");
    anyhow::ensure!(bin.is_file(), "built hi missing at {}", bin.display());
    Ok(bin)
}

async fn run_repro(
    script: &Path,
    hi_binary: &Path,
    worktree: &Path,
    cargo_home: &Path,
    target_dir: &Path,
    failing_test: Option<&[String]>,
    timeout: Duration,
) -> Result<Output> {
    let _ = fsutil::mkdir_0700(cargo_home);
    let _ = fsutil::mkdir_0700(target_dir);
    let mut cmd = tokio::process::Command::new(script);
    cmd.env("HI_BINARY", hi_binary)
        .env("HI_WORKTREE", worktree)
        .env("CARGO_HOME", cargo_home)
        .env("CARGO_TARGET_DIR", target_dir)
        .env("CARGO_TERM_COLOR", "never")
        .current_dir(script.parent().unwrap_or(worktree))
        .kill_on_drop(true)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(args) = failing_test {
        cmd.env("REPRO_CARGO_TEST", args.join(" "));
    }
    timed(cmd, timeout).await
}

fn repro_fail_closed(output: &Output) -> bool {
    output.status.code() == Some(2)
}

pub fn touched_crates(worktree: &Path, base: &str) -> Result<Vec<String>> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(worktree)
        .args(["diff", "--name-only", base, "HEAD"])
        .output()
        .context("git diff --name-only")?;
    let text = String::from_utf8_lossy(&output.stdout);
    let mut crates = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("crates/") else {
            continue;
        };
        let dir = rest.split('/').next().unwrap_or("");
        let pkg = match dir {
            "hi-cli" => "hi",
            other if !other.is_empty() => other,
            _ => continue,
        };
        if !crates.iter().any(|c| c == pkg) {
            crates.push(pkg.to_string());
        }
    }
    Ok(crates)
}

pub fn parse_failing_test(raw: &str) -> Option<Vec<String>> {
    let parts: Vec<String> = raw
        .split_whitespace()
        .filter(|p| !p.is_empty())
        .map(ToString::to_string)
        .collect();
    if parts.is_empty() {
        return None;
    }
    let ok = parts.iter().all(|p| {
        p.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':' | '*'))
    });
    ok.then_some(parts)
}

fn cargo_args(offline: bool, rest: &[&str]) -> Vec<String> {
    let mut args = Vec::new();
    if offline {
        args.push("--offline".into());
    }
    args.extend(rest.iter().map(|s| (*s).to_string()));
    args
}

async fn cargo_offline_warm(worktree: &Path, cargo_home: &Path, target_dir: &Path) -> bool {
    let args = vec![
        "--offline".into(),
        "metadata".into(),
        "--format-version".into(),
        "1".into(),
    ];
    cargo_capture(
        worktree,
        cargo_home,
        target_dir,
        &args,
        Duration::from_secs(30),
    )
    .await
    .map(|o| o.status.success())
    .unwrap_or(false)
}

async fn run_named(
    layer: u8,
    worktree: &Path,
    cargo_home: &Path,
    target_dir: &Path,
    args: Vec<String>,
    timeout: Duration,
) -> Result<LayerReport> {
    let command = format!("cargo {}", args.join(" "));
    let started = Instant::now();
    let output = cargo_capture(worktree, cargo_home, target_dir, &args, timeout).await?;
    Ok(LayerReport {
        layer,
        command,
        passed: output.status.success(),
        skipped: false,
        reason: if output.status.success() {
            None
        } else {
            Some(
                String::from_utf8_lossy(&output.stderr)
                    .chars()
                    .take(800)
                    .collect(),
            )
        },
        duration_ms: elapsed_ms(started),
    })
}

async fn cargo_capture(
    worktree: &Path,
    cargo_home: &Path,
    target_dir: &Path,
    args: &[String],
    timeout: Duration,
) -> Result<Output> {
    let _ = fsutil::mkdir_0700(cargo_home);
    let _ = fsutil::mkdir_0700(target_dir);
    let mut cmd = tokio::process::Command::new("cargo");
    cmd.args(args)
        .current_dir(worktree)
        .env("CARGO_HOME", cargo_home)
        .env("CARGO_TARGET_DIR", target_dir)
        .env("CARGO_TERM_COLOR", "never")
        .kill_on_drop(true)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    timed(cmd, timeout).await
}

async fn timed(mut cmd: tokio::process::Command, timeout: Duration) -> Result<Output> {
    // SAFETY: pre_exec is between fork and exec; setpgid cannot race other threads.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = cmd.spawn().context("spawning cargo/repro")?;
    let pid = child.id().context("cargo/repro pid")?;
    // Parent setpgid is the fallback if the child has not yet run pre_exec.
    unsafe {
        libc::setpgid(pid as i32, pid as i32);
    }
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(out) => out.context("waiting for cargo/repro"),
        Err(_) => {
            spawn::signal_group(pid as i32, libc::SIGTERM);
            tokio::time::sleep(Duration::from_secs(2)).await;
            spawn::signal_group(pid as i32, libc::SIGKILL);
            anyhow::bail!("command timed out after {}s", timeout.as_secs())
        }
    }
}

fn skipped(layer: u8, reason: &str) -> LayerReport {
    LayerReport {
        layer,
        command: String::new(),
        passed: true,
        skipped: true,
        reason: Some(reason.into()),
        duration_ms: 0,
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

fn write_report(incident: &Path, report: &GateReport) -> Result<()> {
    let json = serde_json::to_vec_pretty(report)?;
    fsutil::write_0600(&incident.join("gate.json"), &json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hi_cli_diff_maps_to_package_hi() {
        // touched_crates reads git; this locks the mapping used on the names.
        assert_eq!(
            crate_from_path("crates/hi-cli/src/main.rs").as_deref(),
            Some("hi")
        );
        assert_eq!(
            crate_from_path("crates/hi-harness/src/lib.rs").as_deref(),
            Some("hi-harness")
        );
        assert_eq!(crate_from_path("skills/autoharnessfix/SKILL.md"), None);
    }

    fn crate_from_path(line: &str) -> Option<String> {
        let rest = line.strip_prefix("crates/")?;
        let dir = rest.split('/').next()?;
        Some(match dir {
            "hi-cli" => "hi".into(),
            other => other.into(),
        })
    }

    #[test]
    fn failing_test_rejects_metacharacters() {
        assert!(parse_failing_test("-p hi-harness invariant_holds").is_some());
        assert!(parse_failing_test("-p hi-harness; rm -rf /").is_none());
        assert!(parse_failing_test("").is_none());
    }

    #[test]
    fn touched_crates_uses_base_sha_after_commit() {
        let root = tempfile::tempdir().unwrap();
        let checkout = crate::test_fixture::minimal_checkout(root.path());
        let dest = root.path().join("wt");
        let wt = crate::worktree::add_detached(&checkout, &dest, "HEAD").unwrap();
        let base = wt.base_sha.clone();
        std::fs::write(
            wt.path.join("crates/hi-harness/src/lib.rs"),
            "pub fn patched() {}\n",
        )
        .unwrap();
        crate::worktree::commit_fix(&wt.path, "incident-1-aaaa", "invariant").unwrap();
        assert!(
            touched_crates(&wt.path, "HEAD").unwrap().is_empty(),
            "diff vs HEAD after commit must be empty"
        );
        let vs_base = touched_crates(&wt.path, &base).unwrap();
        assert!(
            vs_base.iter().any(|c| c == "hi-harness"),
            "diff vs base SHA must include hi-harness, got {vs_base:?}"
        );
        crate::worktree::remove(&checkout, &dest);
    }
}
