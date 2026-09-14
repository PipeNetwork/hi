//! Re-exec into `hi-sentinel` when the feature is on. Reads only machine config.

use std::path::PathBuf;

use anyhow::Result;
use hi_liveness::{ENV_HI_BINARY, ENV_ROLE, ENV_SUPERVISED};

use crate::config::{Cli, peek_machine_autoharnessfix, peek_machine_rsi_enabled};

pub(crate) fn skip_exec_for_mode(cli: &Cli) -> bool {
    cli.show_config
        || cli.list_sessions
        || cli.subagent
        || cli.daemon
        || cli.attach.is_some()
        || cli.loops_daemon
        || cli.best_of > 1
        || cli.workflow.is_some()
        || cli.skeptic_review
        || cli.worktree
        || cli.sentinel.sentinel_restore_checkpoint.is_some()
}

pub(crate) fn rsi_off_limits(cli: &Cli) -> bool {
    if cli.rsi_managed || cli.rsi {
        return true;
    }
    if std::env::var_os("HI_RUNTIME_DESCRIPTOR").is_some() {
        return true;
    }
    if let Ok(value) = std::env::var("HI_RSI_ENABLED") {
        let trimmed = value.trim().to_ascii_lowercase();
        match trimmed.as_str() {
            "1" | "true" | "on" | "yes" => return true,
            "" | "0" | "false" | "off" | "no" => {}
            _ => return true,
        }
    }
    peek_machine_rsi_enabled()
}

pub(crate) fn sentinel_requested(cli: &Cli) -> bool {
    if cli.sentinel.no_autoharnessfix {
        return false;
    }
    cli.sentinel.autoharnessfix
        || cli.sentinel.autoharnessfix_apply
        || peek_machine_autoharnessfix().enabled
}

pub(crate) fn sentinel_flag_error(cli: &Cli) -> Option<String> {
    if cli.sentinel.autoharnessfix && (cli.rsi_managed || cli.rsi) {
        return Some("--autoharnessfix cannot be used with --rsi-managed or --rsi".into());
    }
    if cli.sentinel.autoharnessfix_apply
        && !cli.sentinel.autoharnessfix
        && !peek_machine_autoharnessfix().enabled
    {
        return Some(
            "--autoharnessfix-apply requires --autoharnessfix or machine [autoharnessfix] enabled = true".into(),
        );
    }
    #[cfg(not(unix))]
    if cli.sentinel.autoharnessfix || cli.sentinel.autoharnessfix_apply {
        return Some("hi --autoharnessfix is Unix-only".into());
    }
    None
}

pub(crate) fn find_sentinel_binary() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("HI_SENTINEL_BINARY") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let sibling = dir.join("hi-sentinel");
        if sibling.exists() {
            return Some(sibling);
        }
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join("hi-sentinel");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub fn maybe_exec_into_sentinel(cli: &Cli) -> Result<()> {
    if std::env::var_os(ENV_SUPERVISED).is_some() {
        return Ok(());
    }
    if std::env::var_os(ENV_ROLE).is_some_and(|value| value == "repair" || value == "restore") {
        return Ok(());
    }
    if skip_exec_for_mode(cli) {
        return Ok(());
    }
    if rsi_off_limits(cli) {
        return Ok(());
    }
    if !sentinel_requested(cli) {
        return Ok(());
    }
    #[cfg(not(unix))]
    {
        eprintln!("hi --autoharnessfix is Unix-only");
        std::process::exit(2);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let Some(sentinel) = find_sentinel_binary() else {
            eprintln!(
                "hi --autoharnessfix requires the hi-sentinel binary; build with cargo build -p hi -p hi-sentinel or re-run scripts/install.sh"
            );
            std::process::exit(2);
        };
        let mut args = std::env::args_os();
        let _argv0 = args.next();
        let err = std::process::Command::new(&sentinel)
            .arg("--")
            .args(args)
            .env(ENV_HI_BINARY, std::env::current_exe()?)
            .exec();
        anyhow::bail!("exec hi-sentinel: {err}");
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RestoreCheckpointAction {
    Skip,
    Fail { message: &'static str },
    Restore { id: String },
}

pub(crate) fn restore_checkpoint_action(cli: &Cli) -> RestoreCheckpointAction {
    let restore_role = std::env::var(ENV_ROLE).is_ok_and(|value| value == "restore");
    match (
        cli.sentinel.sentinel_restore_checkpoint.as_deref(),
        restore_role,
    ) {
        (None, false) => RestoreCheckpointAction::Skip,
        (None, true) => RestoreCheckpointAction::Fail {
            message: "HI_SENTINEL_ROLE=restore requires --sentinel-restore-checkpoint",
        },
        (Some(_), false) => RestoreCheckpointAction::Fail {
            message: "hidden --sentinel-restore-checkpoint requires HI_SENTINEL_ROLE=restore",
        },
        (Some(id), true) => RestoreCheckpointAction::Restore { id: id.to_string() },
    }
}

pub async fn maybe_restore_checkpoint(cli: &Cli) -> Result<()> {
    match restore_checkpoint_action(cli) {
        RestoreCheckpointAction::Skip => Ok(()),
        RestoreCheckpointAction::Fail { message } => {
            eprintln!("{message}");
            std::process::exit(2);
        }
        RestoreCheckpointAction::Restore { id } => {
            let Some(target) = cli.review_target.as_deref() else {
                eprintln!("--sentinel-restore-checkpoint requires --review-target DIR");
                std::process::exit(2);
            };
            crate::review_target::chdir_to_review_target(target)?;
            let (workspace_root, state_root) = crate::paths::resolve_runtime_roots()?;
            let n =
                hi_tools::checkpoint::restore_with_state(&workspace_root, &id, &state_root).await?;
            println!("restored {n} path(s)");
            std::process::exit(0);
        }
    }
}
