//! Session JSONL path resolution for the Pipe frontend.

use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::Result;

use crate::config::Cli;
use crate::paths;

pub fn resolve_session_path(cli: &Cli) -> Result<Option<PathBuf>> {
    if let Some(path) = &cli.session_file {
        return Ok(Some(path.clone()));
    }
    if cli.no_save {
        return Ok(None);
    }
    if let Some(id) = &cli.resume {
        return Ok(Some(paths::session_path(id)?));
    }
    if cli.cont {
        if let Some(path) = pick_continue_session()? {
            return Ok(Some(path));
        }
        eprintln!("\x1b[33mno previous session; starting a new one\x1b[0m");
    }
    Ok(Some(paths::new_session_path()?))
}

fn pick_continue_session() -> Result<Option<PathBuf>> {
    let Some(root) = paths::data_root() else {
        return Ok(paths::latest_session());
    };
    let digest = paths::cwd_digest();
    let latest = paths::latest_session();
    crate::roster::pick_unfinished(
        &root,
        &digest,
        latest.as_deref(),
        std::io::stdin().is_terminal(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Cli;
    use clap::Parser;
    use hi_tools::{PlanStatus, PlanStep};

    #[test]
    fn session_path_prefers_explicit_file() {
        let cli = Cli::try_parse_from(["hi", "--session-file", "/tmp/explicit.jsonl"]).unwrap();
        let path = resolve_session_path(&cli).unwrap();
        assert_eq!(
            path.as_deref(),
            Some(std::path::Path::new("/tmp/explicit.jsonl"))
        );
    }

    #[test]
    fn session_path_no_save_skips_persistence() {
        let cli = Cli::try_parse_from(["hi", "--no-save"]).unwrap();
        assert_eq!(resolve_session_path(&cli).unwrap(), None);
    }

    #[test]
    fn session_path_resume_id_is_used() {
        let cli = Cli::try_parse_from(["hi", "--resume", "abc-123"]).unwrap();
        let path = resolve_session_path(&cli).unwrap().expect("path");
        assert!(
            path.ends_with("abc-123.jsonl"),
            "unexpected resume path {}",
            path.display()
        );
    }

    #[test]
    fn continue_prefers_open_plan_over_newer_empty_review() {
        let _cwd = crate::CWD_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let prev_cwd = std::env::current_dir().unwrap();
        let prev_xdg = std::env::var_os("XDG_DATA_HOME");
        std::env::set_current_dir(tmp.path()).unwrap();
        let xdg = tmp.path().join("xdg");
        unsafe {
            std::env::set_var("XDG_DATA_HOME", &xdg);
        }
        let digest = paths::cwd_digest();
        let sessions = xdg
            .join("hi")
            .join("projects")
            .join(&digest)
            .join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        let open = sessions.join("old-open.jsonl");
        let mut session = hi_harness::JsonlSession::create(&open).unwrap();
        session
            .record_messages(&[hi_ai::Message::user("finish pagination")])
            .unwrap();
        session
            .record_plan(&[
                PlanStep {
                    title: "Forward HISTORY pagination".into(),
                    status: PlanStatus::Done,
                },
                PlanStep {
                    title: "Next".into(),
                    status: PlanStatus::Pending,
                },
            ])
            .unwrap();
        drop(session);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let newer = sessions.join("new-review.jsonl");
        let mut session = hi_harness::JsonlSession::create(&newer).unwrap();
        session
            .record_messages(&[hi_ai::Message::user("unrelated review")])
            .unwrap();
        drop(session);

        let picked = pick_continue_session().unwrap().expect("unfinished");
        assert_eq!(picked, open);

        std::env::set_current_dir(prev_cwd).unwrap();
        unsafe {
            match prev_xdg {
                Some(value) => std::env::set_var("XDG_DATA_HOME", value),
                None => std::env::remove_var("XDG_DATA_HOME"),
            }
        }
    }
}
