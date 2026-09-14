//! Interactive `/autoharnessfix` helpers shared by the TUI and the plain REPL.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use hi_liveness::{ENV_GENERATION, ENV_HI_BINARY, ENV_ROLE, ENV_SUPERVISED, env_flag_on};

use crate::args::find_on_path;
use crate::budget;
use crate::classify::{BugKind, Class, Confidence};
use crate::config::{self, RepairConfig};
use crate::history;
use crate::incident;
use crate::ipc;
use crate::paths;
use crate::{ENV_BINARY, ENV_CHECKOUT};

const USAGE: &str = "use /autoharnessfix on|off|status|diagnose|history|repair";
const RESTART_LINE: &str = "Restarting under Sentinel…";
const ENABLED_RESTART: &str = "Sentinel enabled; restart hi";
const MISSING_BUILD_HINT: &str = "Build it with `cargo build --release` (hi and hi-sentinel) or `cargo build -p hi-sentinel --release`, then restart. `/autoharnessfix off` disables.";

#[derive(Debug)]
pub enum SlashOutcome {
    Message(String),
    ExecSupervisor { session_file: PathBuf },
    ExecRepair { incident: PathBuf },
}

pub fn dispatch(
    arg: &str,
    session_path: Option<&Path>,
    no_save: bool,
    rsi_blocked: Option<&str>,
    workspace: &Path,
) -> Result<SlashOutcome> {
    let verb = arg
        .split_whitespace()
        .next()
        .unwrap_or("status")
        .to_ascii_lowercase();
    match verb.as_str() {
        "" | "status" => Ok(SlashOutcome::Message(status_text())),
        "on" => enable(session_path, no_save, rsi_blocked),
        "off" => {
            let path = config::set_machine_enabled(false)?;
            Ok(SlashOutcome::Message(format!(
                "Sentinel disabled in {} (this process stays as-is until exit)",
                path.display()
            )))
        }
        "diagnose" => Ok(SlashOutcome::Message(diagnose(workspace, session_path)?)),
        "history" => Ok(SlashOutcome::Message(history_text())),
        "repair" => repair(workspace, session_path),
        _ => Ok(SlashOutcome::Message(USAGE.into())),
    }
}

pub fn status_text() -> String {
    let supervised = std::env::var(ENV_SUPERVISED)
        .ok()
        .as_deref()
        .is_some_and(env_flag_on);
    let generation: u32 = std::env::var(ENV_GENERATION)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let machine = config::peek_machine().unwrap_or_default();
    let checkout = machine
        .checkout
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "(not configured)".into());
    let repair_cfg = RepairConfig::from_machine_and_env();
    let remaining = repair_cfg
        .max_repairs_per_session
        .saturating_sub(generation);
    let applies = budget::applies_last_hour(
        &paths::apply_log_path(&paths::state_dir()),
        hi_liveness::unix_ms(),
    );
    let applies_left = repair_cfg
        .max_modifications_per_hour
        .saturating_sub(applies);
    let last = history::tail(&paths::history_path(&paths::state_dir()), 1)
        .into_iter()
        .next()
        .map(|line| format!("{} ({}/{})", line.id, line.class, line.kind))
        .unwrap_or_else(|| "(none)".into());
    format!(
        "sentinel: {} (generation {generation})\ncheckout: {checkout}\nlast incident: {last}\nrepairs remaining this session: {remaining}\napplies remaining this hour: {applies_left}",
        if supervised { "on" } else { "off" }
    )
}

pub fn history_text() -> String {
    let state = paths::state_dir();
    let incidents = history::tail(&paths::history_path(&state), 20);
    let applies = budget::tail_apply(&paths::apply_log_path(&state), 20);
    if incidents.is_empty() && applies.is_empty() {
        return "no Sentinel incidents".into();
    }
    let mut out = Vec::new();
    out.push("incidents:".into());
    if incidents.is_empty() {
        out.push("  (none)".into());
    } else {
        for line in incidents {
            out.push(format!("  {}  {}  {}", line.id, line.class, line.kind));
        }
    }
    out.push("applies:".into());
    if applies.is_empty() {
        out.push("  (none)".into());
    } else {
        for line in applies {
            out.push(format!("  {}", line.id));
        }
    }
    out.join("\n")
}

fn enable(
    session_path: Option<&Path>,
    no_save: bool,
    rsi_blocked: Option<&str>,
) -> Result<SlashOutcome> {
    if let Some(reason) = rsi_blocked {
        bail!("{reason}");
    }
    let path = config::set_machine_enabled(true)?;
    let _ = config::ensure_machine_checkout();
    if no_save {
        return Ok(SlashOutcome::Message(format!(
            "{ENABLED_RESTART} (wrote {})",
            path.display()
        )));
    }
    let Some(session_file) = session_path else {
        return Ok(SlashOutcome::Message(format!(
            "{ENABLED_RESTART} (wrote {})",
            path.display()
        )));
    };
    if already_inside_sentinel() {
        return Ok(SlashOutcome::Message(format!(
            "Sentinel already supervising this process ({})",
            path.display()
        )));
    }
    if find_sentinel_binary().is_none() {
        return Ok(SlashOutcome::Message(format!(
            "Sentinel enabled in {} but {} is missing.\n{}",
            path.display(),
            sibling_hint(),
            MISSING_BUILD_HINT
        )));
    }
    Ok(SlashOutcome::ExecSupervisor {
        session_file: session_file.to_path_buf(),
    })
}

pub fn diagnose(workspace: &Path, session_path: Option<&Path>) -> Result<String> {
    if let Some(runtime) = ipc::runtime_dir_from_env() {
        ipc::write_request(&runtime, ipc::REQUEST_DIAGNOSE, ipc::REQUEST_DIAGNOSE_DONE)
            .context("writing diagnose request")?;
        Ok("diagnose requested; Sentinel will write a bundle (see /autoharnessfix history)".into())
    } else {
        let dir = incident::write_local_snapshot(workspace, session_path, crash_dir().as_deref())
            .context("writing diagnose snapshot")?;
        Ok(format!("diagnose bundle: {}", dir.display()))
    }
}

fn repair(workspace: &Path, session_path: Option<&Path>) -> Result<SlashOutcome> {
    if ipc::runtime_dir_from_env().is_some() {
        let runtime = ipc::runtime_dir_from_env().expect("supervised runtime");
        ipc::write_request(&runtime, ipc::REQUEST_REPAIR, ipc::REQUEST_REPAIR_DONE)
            .context("writing repair request")?;
        return Ok(SlashOutcome::Message(
            "repair requested; this session stays up. Result lands in /autoharnessfix history (apply needs [autoharnessfix] apply = true, or a restart)."
                .into(),
        ));
    }
    let dir = incident::write_local_snapshot(workspace, session_path, crash_dir().as_deref())
        .context("writing diagnose snapshot for repair")?;
    Ok(SlashOutcome::ExecRepair { incident: dir })
}

fn crash_dir() -> Option<PathBuf> {
    std::env::var_os(hi_liveness::ENV_CRASH_DIR)
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".hi/crash")))
}

pub fn restart_line() -> &'static str {
    RESTART_LINE
}

pub fn find_sentinel_binary() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(ENV_BINARY) {
        let path = PathBuf::from(path);
        return path.exists().then_some(path);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let sibling = dir.join("hi-sentinel");
        if sibling.exists() {
            return Some(sibling);
        }
    }
    find_on_path("hi-sentinel")
}

fn sibling_hint() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("hi-sentinel")))
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "hi-sentinel".into())
}

fn missing_sentinel() -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "hi --autoharnessfix requires {}; {MISSING_BUILD_HINT}",
            sibling_hint()
        ),
    )
}

fn already_inside_sentinel() -> bool {
    std::env::var(ENV_SUPERVISED)
        .ok()
        .as_deref()
        .is_some_and(env_flag_on)
        || std::env::var(ENV_ROLE).is_ok_and(|role| role == "harness")
}

/// Drop the empty pre-opened crash marker before `exec(2)`. The fd is CLOEXEC,
/// so the sentinel PID would otherwise leave a live empty `last-crash-{pid}-*.bin`.
pub fn unlink_empty_crash_markers_for_current_pid() {
    let Some(dir) = crash_dir() else {
        return;
    };
    let prefix = format!("last-crash-{}-", std::process::id());
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with(&prefix)
            && name.ends_with(".bin")
            && entry.metadata().map(|m| m.len() == 0).unwrap_or(false)
        {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Pin `--session-file` so a wrap/re-exec continues the same JSONL.
pub fn ensure_session_file_arg(args: &mut Vec<std::ffi::OsString>, session_file: &Path) {
    if args.iter().any(|arg| arg == "--session-file") {
        return;
    }
    args.push("--session-file".into());
    args.push(session_file.into());
}

/// Replace this process with `hi-sentinel`. Caller must restore the tty first.
#[cfg(unix)]
pub fn exec_supervisor(session_file: &Path) -> io::Error {
    use std::os::unix::process::CommandExt;
    let Some(sentinel) = find_sentinel_binary() else {
        return missing_sentinel();
    };
    let mut args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    ensure_session_file_arg(&mut args, session_file);
    unlink_empty_crash_markers_for_current_pid();
    let mut command = Command::new(&sentinel);
    command.arg("--").args(&args);
    if let Ok(exe) = std::env::current_exe() {
        command.env(ENV_HI_BINARY, exe);
    }
    command.exec()
}

#[cfg(not(unix))]
pub fn exec_supervisor(_session_file: &Path) -> io::Error {
    io::Error::other("hi-sentinel is Unix-only")
}

#[cfg(unix)]
pub fn exec_repair(incident: &Path) -> io::Error {
    use std::os::unix::process::CommandExt;
    let Some(sentinel) = find_sentinel_binary() else {
        return missing_sentinel();
    };
    let mut command = Command::new(&sentinel);
    command.arg("repair").arg("--incident").arg(incident);
    if let Some(checkout) = config::peek_machine().and_then(|section| section.checkout) {
        command.arg("--checkout").arg(checkout);
    } else if let Some(checkout) = std::env::var_os(ENV_CHECKOUT) {
        command.arg("--checkout").arg(checkout);
    }
    command.exec()
}

#[cfg(not(unix))]
pub fn exec_repair(_incident: &Path) -> io::Error {
    io::Error::other("hi-sentinel is Unix-only")
}

pub fn manual_class() -> Class {
    Class::HarnessBug {
        kind: BugKind::Invariant,
        confidence: Confidence::High,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_session_file_arg_is_idempotent() {
        let mut args = Vec::new();
        let path = PathBuf::from("/tmp/sess.jsonl");
        ensure_session_file_arg(&mut args, &path);
        ensure_session_file_arg(&mut args, &path);
        assert_eq!(
            args,
            vec![
                std::ffi::OsString::from("--session-file"),
                std::ffi::OsString::from("/tmp/sess.jsonl"),
            ]
        );
    }

    #[test]
    fn on_inside_harness_role_does_not_exec() {
        let _lock = crate::ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let previous_config = std::env::var_os("XDG_CONFIG_HOME");
        let previous_role = std::env::var_os(ENV_ROLE);
        let previous_sup = std::env::var_os(ENV_SUPERVISED);
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", tmp.path());
            std::env::set_var(ENV_ROLE, "harness");
            std::env::remove_var(ENV_SUPERVISED);
        }
        let session = tmp.path().join("session.jsonl");
        std::fs::write(&session, b"").unwrap();
        let out = dispatch("on", Some(&session), false, None, tmp.path()).unwrap();
        unsafe {
            match previous_config {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
            match previous_role {
                Some(value) => std::env::set_var(ENV_ROLE, value),
                None => std::env::remove_var(ENV_ROLE),
            }
            match previous_sup {
                Some(value) => std::env::set_var(ENV_SUPERVISED, value),
                None => std::env::remove_var(ENV_SUPERVISED),
            }
        }
        match out {
            SlashOutcome::Message(text) => {
                assert!(text.contains("already supervising"), "{text}");
            }
            other => panic!("expected stay-inside-sentinel, got {other:?}"),
        }
    }

    #[test]
    fn unknown_verb_is_usage() {
        let dir = tempfile::tempdir().unwrap();
        let out = dispatch("nope", None, true, None, dir.path()).unwrap();
        match out {
            SlashOutcome::Message(text) => assert!(text.contains("on|off|status")),
            _ => panic!("expected usage message"),
        }
    }

    #[test]
    fn on_without_binary_stays_in_session() {
        let _lock = crate::ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let previous_config = std::env::var_os("XDG_CONFIG_HOME");
        let previous_bin = std::env::var_os(ENV_BINARY);
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", tmp.path());
            std::env::set_var(ENV_BINARY, tmp.path().join("no-such-hi-sentinel"));
        }
        let session = tmp.path().join("session.jsonl");
        std::fs::write(&session, b"").unwrap();
        let out = dispatch("on", Some(&session), false, None, tmp.path()).unwrap();
        unsafe {
            match previous_config {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
            match previous_bin {
                Some(value) => std::env::set_var(ENV_BINARY, value),
                None => std::env::remove_var(ENV_BINARY),
            }
        }
        match out {
            SlashOutcome::Message(text) => {
                assert!(text.contains("missing"), "{text}");
                assert!(text.contains("cargo build"), "{text}");
            }
            other => panic!("expected stay-in-session message, got {other:?}"),
        }
    }

    #[test]
    fn on_with_no_save_does_not_exec() {
        let _lock = crate::ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("XDG_CONFIG_HOME");
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", tmp.path());
        }
        let session = tmp.path().join("session.jsonl");
        let out = dispatch("on", Some(&session), true, None, tmp.path()).unwrap();
        unsafe {
            match previous {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
        match out {
            SlashOutcome::Message(text) => {
                assert!(text.contains("restart hi"), "{text}");
            }
            other => panic!("expected persist-only, got {other:?}"),
        }
    }

    #[test]
    fn supervised_diagnose_does_not_wait() {
        let _lock = crate::ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let hb = tmp.path().join("heartbeat.json");
        std::fs::write(&hb, b"{}").unwrap();
        let previous = std::env::var_os(hi_liveness::ENV_HEARTBEAT);
        unsafe {
            std::env::set_var(hi_liveness::ENV_HEARTBEAT, &hb);
        }
        let started = std::time::Instant::now();
        let text = diagnose(tmp.path(), None);
        let elapsed = started.elapsed();
        unsafe {
            match previous {
                Some(value) => std::env::set_var(hi_liveness::ENV_HEARTBEAT, value),
                None => std::env::remove_var(hi_liveness::ENV_HEARTBEAT),
            }
        }
        let text = text.unwrap();
        assert!(text.contains("diagnose requested"), "{text}");
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "diagnose blocked the caller for {elapsed:?}"
        );
    }

    #[test]
    fn history_text_includes_apply_log() {
        let _lock = crate::ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let previous = std::env::var_os(crate::ENV_STATE_DIR);
        unsafe {
            std::env::set_var(crate::ENV_STATE_DIR, tmp.path());
        }
        crate::fsutil::mkdir_0700(tmp.path()).unwrap();
        crate::history::append_raw(
            &crate::paths::history_path(tmp.path()),
            "incident-1-aaaa",
            "diagnose",
            "snapshot",
        )
        .unwrap();
        crate::budget::append_apply(
            &crate::paths::apply_log_path(tmp.path()),
            "incident-1-aaaa",
            1,
        )
        .unwrap();
        let text = history_text();
        unsafe {
            match previous {
                Some(value) => std::env::set_var(crate::ENV_STATE_DIR, value),
                None => std::env::remove_var(crate::ENV_STATE_DIR),
            }
        }
        assert!(text.contains("incidents:"), "{text}");
        assert!(text.contains("incident-1-aaaa"), "{text}");
        assert!(text.contains("applies:"), "{text}");
    }
}
