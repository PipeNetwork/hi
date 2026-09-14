//! Forensic bundle writer. Incident dirs are 0700, files 0600, secrets redacted.

use std::fs;
use std::io::{self, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use hi_liveness::{Heartbeat, unix_ms};
use hi_secrets::{SECRET_ENV_NAMES, redact_secrets};
use serde::Serialize;

use crate::classify::Class;
use crate::config::SupervisorConfig;
use crate::fsutil;
use crate::history;
use crate::paths;

const REPRO_SCRIPT: &str = r#"#!/usr/bin/env bash
set -euo pipefail
# Isolated scratch. Layer 0 sets HI_BINARY to a worktree-built hi (HEAD, then repair).
ROOT="$(cd "$(dirname "$0")" && pwd)"
HI_BINARY="${HI_BINARY:?set HI_BINARY to the hi under test}"
if [[ -n "${REPRO_CARGO_TEST:-}" ]]; then
  cd "${HI_WORKTREE:?}"
  exec cargo test ${REPRO_CARGO_TEST}
fi
if [[ -d "$ROOT/user-project-copy" ]]; then
  cd "$ROOT/user-project-copy"
  echo "no deterministic harness repro; fail closed" >&2
  exit 2
fi
echo "no isolated repro fixture; fail closed" >&2
exit 2
"#;

#[derive(Serialize)]
struct IncidentDoc {
    schema_version: u32,
    id: String,
    created_unix_ms: u64,
    class: String,
    kind: String,
    duration_ms: u64,
    generation: u32,
    hi_version: String,
    hi_binary: String,
    hi_binary_blake3: String,
    checkout: Option<CheckoutDoc>,
    user_workspace: String,
    session_path: Option<String>,
    last_state: Option<String>,
    last_tool: Option<String>,
    pre_checkpoint: Option<String>,
    original_argv: Vec<String>,
    oneshot: bool,
    plain: bool,
    reproduction: String,
    redacted: bool,
}

#[derive(Serialize)]
struct CheckoutDoc {
    path: String,
    head_sha: String,
    dirty: bool,
}

pub struct Bundle {
    pub dir: PathBuf,
}

pub fn write_bundle(
    cfg: &SupervisorConfig,
    class: &Class,
    runtime: &Path,
    heartbeat: Option<&Heartbeat>,
    duration: Duration,
) -> Result<Bundle> {
    let incidents = paths::incidents_dir(&cfg.state_dir);
    fsutil::mkdir_0700(&incidents)?;
    let id = allocate_id(&cfg.state_dir)?;
    let dir = incidents.join(&id);
    fsutil::mkdir_0700(&dir)?;

    copy_redacted_if_exists(&runtime.join("events.jsonl"), &dir.join("events.jsonl"))?;
    if let Some(hb) = heartbeat {
        let json = serde_json::to_vec_pretty(hb).unwrap_or_default();
        fsutil::write_0600(
            &dir.join("heartbeat-last.json"),
            redact_secrets(&String::from_utf8_lossy(&json)).as_bytes(),
        )?;
    } else {
        copy_redacted_if_exists(
            &runtime.join("heartbeat.json"),
            &dir.join("heartbeat-last.json"),
        )?;
    }
    copy_redacted_if_exists(
        &runtime.join("turn-intent.json"),
        &dir.join("turn-intent.json"),
    )?;
    copy_redacted_if_exists(&runtime.join("panic.txt"), &dir.join("panic.txt"))?;
    copy_crash_dir(&runtime.join("crash"), &dir.join("crash"))?;
    write_process_tree(&dir.join("process-tree.txt"), heartbeat)?;
    write_environment(&dir.join("environment.txt"))?;
    write_git_status(&dir.join("git-status-user.txt"), &cfg.workspace)?;
    if let Some(checkout) = &cfg.checkout {
        write_git_status(&dir.join("git-status-checkout.txt"), checkout)?;
    }
    copy_redacted_if_exists(
        &paths::known_good_path(&cfg.state_dir),
        &dir.join("known-good.json"),
    )?;
    if let Some(session) = heartbeat.and_then(|h| h.session_path.as_ref()) {
        copy_redacted_if_exists(Path::new(session), &dir.join("transcript.jsonl"))?;
    }
    fsutil::write_0600(
        &dir.join("stdout.log"),
        b"(tui owns stdout; not captured)\n",
    )?;
    let panic = runtime.join("panic.txt");
    if panic.exists() {
        copy_redacted_if_exists(&panic, &dir.join("stderr.log"))?;
    } else {
        fsutil::write_0600(&dir.join("stderr.log"), b"")?;
    }

    let repro_dir = dir.join("repro");
    fsutil::mkdir_0700(&repro_dir)?;
    let repro = repro_dir.join("reproduction.sh");
    fsutil::write_0600(&repro, REPRO_SCRIPT.as_bytes())?;
    fsutil::chmod_0700_file(&repro)?;
    fsutil::write_0600(
        &repro_dir.join("README"),
        b"Isolated reproduction. Never exec the live project or installed binary.\n",
    )?;

    let checkout = cfg.checkout.as_ref().and_then(|path| {
        crate::checkout::validate(path).ok().map(|v| CheckoutDoc {
            path: v.path.display().to_string(),
            head_sha: v.head_sha,
            dirty: v.dirty,
        })
    });
    let hi_binary = cfg.hi_binary.display().to_string();
    let hi_binary_blake3 = blake3_file(&cfg.hi_binary).unwrap_or_default();
    let last_state = heartbeat.map(|h| state_slug(h.state));
    let last_tool = heartbeat.and_then(|h| h.last_tool.clone());
    let pre_checkpoint = heartbeat.and_then(|h| h.pre_checkpoint.clone());
    let session_path = heartbeat.and_then(|h| h.session_path.clone());
    let plain = cfg.original_argv.iter().any(|a| a == "--plain");
    let doc = IncidentDoc {
        schema_version: 1,
        id: id.clone(),
        created_unix_ms: unix_ms(),
        class: class.class_slug().into(),
        kind: class.kind_slug().into(),
        duration_ms: duration.as_millis() as u64,
        generation: cfg.generation,
        hi_version: env!("CARGO_PKG_VERSION").into(),
        hi_binary,
        hi_binary_blake3,
        checkout,
        user_workspace: cfg.workspace.display().to_string(),
        session_path,
        last_state,
        last_tool,
        pre_checkpoint,
        original_argv: cfg.original_argv.clone(),
        oneshot: plain,
        plain,
        reproduction: "repro/reproduction.sh".into(),
        redacted: true,
    };
    let json = serde_json::to_vec_pretty(&doc)?;
    fsutil::write_0600(&dir.join("incident.json"), &json)?;
    history::append(&paths::history_path(&cfg.state_dir), &id, class)?;
    fsutil::chmod_0700(&dir)?;
    let _ = id;
    Ok(Bundle { dir })
}

fn allocate_id(state: &Path) -> io::Result<String> {
    fsutil::mkdir_0700(state)?;
    let path = paths::next_id_path(state);
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)?;
    fsutil::chmod_0600(&path)?;
    // SAFETY: exclusive flock on a file we created in the private state dir.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    let mut body = String::new();
    use std::io::Read;
    let mut file = file;
    file.read_to_string(&mut body)?;
    let next = body.trim().parse::<u64>().unwrap_or(0).saturating_add(1);
    file.set_len(0)?;
    {
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0))?;
    }
    writeln!(file, "{next}")?;
    file.flush()?;
    unsafe {
        libc::flock(file.as_raw_fd(), libc::LOCK_UN);
    }
    let nonce = (unix_ms() ^ u64::from(std::process::id())) as u16;
    Ok(format!("incident-{next}-{nonce:04x}"))
}

fn copy_redacted_if_exists(from: &Path, to: &Path) -> io::Result<()> {
    if !from.exists() {
        return Ok(());
    }
    let bytes = fs::read(from)?;
    let text = String::from_utf8_lossy(&bytes);
    fsutil::write_0600(to, redact_secrets(&text).as_bytes())
}

fn copy_crash_dir(from: &Path, to: &Path) -> io::Result<()> {
    if !from.exists() {
        return Ok(());
    }
    fsutil::mkdir_0700(to)?;
    let Ok(entries) = fs::read_dir(from) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        let dest = to.join(entry.file_name());
        if entry.path().is_file() {
            copy_redacted_if_exists(&entry.path(), &dest)?;
        }
    }
    Ok(())
}

fn write_process_tree(path: &Path, heartbeat: Option<&Heartbeat>) -> io::Result<()> {
    let output = std::process::Command::new("ps")
        .args(["-ax", "-o", "pid,ppid,pgid,stat,command"])
        .output();
    let mut body = String::new();
    if let Some(hb) = heartbeat {
        body.push_str(&format!(
            "current_tool_pgid={:?} child_pgids={:?}\n",
            hb.current_tool_pgid, hb.child_pgids
        ));
    }
    if let Ok(output) = output {
        body.push_str(&redact_secrets(&String::from_utf8_lossy(&output.stdout)));
    }
    fsutil::write_0600(path, body.as_bytes())
}

fn write_environment(path: &Path) -> io::Result<()> {
    let mut lines = Vec::new();
    for key in [
        "TERM",
        "COLORTERM",
        "HI_SANDBOX",
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "LANG",
    ] {
        if SECRET_ENV_NAMES.contains(&key) {
            continue;
        }
        if let Ok(value) = std::env::var(key) {
            let value = if key == "PATH" {
                truncate(&value, 512)
            } else {
                value
            };
            lines.push(format!("{key}={}", redact_secrets(&value)));
        }
    }
    for (key, value) in std::env::vars() {
        if key.contains("API_KEY") || key.contains("TOKEN") || key.contains("SECRET") {
            continue;
        }
        if key.starts_with("HI_SENTINEL_") {
            lines.push(format!("{key}={}", redact_secrets(&value)));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        lines.push(format!("cwd={}", cwd.display()));
    }
    lines.push(format!("uid={}", unsafe { libc::getuid() }));
    if let Ok(output) = std::process::Command::new("uname").arg("-a").output() {
        let uname = String::from_utf8_lossy(&output.stdout);
        lines.push(redact_secrets(uname.trim()).into_owned());
    }
    fsutil::write_0600(path, lines.join("\n").as_bytes())
}

fn write_git_status(path: &Path, dir: &Path) -> io::Result<()> {
    let mut body = String::new();
    for args in [
        vec!["-C", &dir.display().to_string(), "rev-parse", "HEAD"],
        vec!["-C", &dir.display().to_string(), "status"],
    ] {
        if let Ok(output) = std::process::Command::new("git").args(&args).output() {
            body.push_str(&redact_secrets(&String::from_utf8_lossy(&output.stdout)));
            body.push_str(&redact_secrets(&String::from_utf8_lossy(&output.stderr)));
        }
    }
    fsutil::write_0600(path, body.as_bytes())
}

fn blake3_file(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    Some(blake3::hash(&bytes).to_hex().to_string())
}

fn state_slug(state: hi_liveness::HarnessState) -> String {
    serde_json::to_string(&state)
        .unwrap_or_default()
        .trim_matches('"')
        .to_string()
}

fn truncate(value: &str, max: usize) -> String {
    let mut out: String = value.chars().take(max).collect();
    if out.chars().count() < value.chars().count() {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_liveness::HarnessState;

    #[test]
    fn last_state_uses_snake_case() {
        assert_eq!(state_slug(HarnessState::AwaitingUser), "awaiting_user");
        assert_eq!(state_slug(HarnessState::ExecutingTool), "executing_tool");
    }

    #[test]
    fn truncate_does_not_split_multibyte_chars() {
        let value = "é".repeat(600);
        let out = truncate(&value, 512);
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), 513);
    }
}
