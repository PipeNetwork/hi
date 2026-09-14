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

fn already_supervised() -> bool {
    std::env::var_os(ENV_SUPERVISED).is_some()
        || std::env::var(ENV_ROLE)
            .is_ok_and(|role| matches!(role.as_str(), "harness" | "repair" | "restore"))
}

pub fn maybe_exec_into_sentinel(cli: &Cli) -> Result<()> {
    if already_supervised() {
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
        let Some(sentinel) = hi_sentinel::find_sentinel_binary() else {
            let next_to = std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(|dir| dir.join("hi-sentinel")));
            eprintln!(
                "hi --autoharnessfix requires the hi-sentinel binary (looked for {}). cargo build --release, or cargo build -p hi-sentinel --release, or scripts/install.sh. /autoharnessfix off disables.",
                next_to
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "hi-sentinel".into())
            );
            std::process::exit(2);
        };
        let mut args: Vec<_> = std::env::args_os().skip(1).collect();
        if !cli.no_save {
            match crate::session_files::resolve_session_path(cli) {
                Ok(Some(path)) => hi_sentinel::ensure_session_file_arg(&mut args, &path),
                Ok(None) => {}
                Err(err) => {
                    eprintln!("hi --autoharnessfix: {err:#}");
                    std::process::exit(2);
                }
            }
        }
        hi_sentinel::unlink_empty_crash_markers_for_current_pid();
        let err = std::process::Command::new(&sentinel)
            .arg("--")
            .args(&args)
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
