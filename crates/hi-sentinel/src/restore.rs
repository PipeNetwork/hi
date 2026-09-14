//! User-project restore via known-good `hi --sentinel-restore-checkpoint`.
//! Always conservative — `--autoharnessfix-apply` does not skip the prompt.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use hi_liveness::{ENV_ROLE, ENV_SUPERVISED};

use crate::spawn;

#[derive(Clone, Debug)]
pub struct RestoreRequest {
    pub hi_binary: PathBuf,
    pub workspace: PathBuf,
    pub pre_checkpoint: Option<String>,
    pub stdin_is_tty: bool,
    pub answer: Option<String>,
    /// Tests inject presence; `None` probes the binary.
    pub flag_present: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestoreOutcome {
    Restored { paths: usize },
    Skipped { reason: String },
}

pub fn maybe_restore_user_project(req: &RestoreRequest) -> RestoreOutcome {
    let Some(id) = req.pre_checkpoint.as_deref().filter(|id| !id.is_empty()) else {
        return RestoreOutcome::Skipped {
            reason: "no pre-turn checkpoint".into(),
        };
    };
    spawn::prepare_interactive_prompt();
    let question = format!("Restore user workspace to pre-turn checkpoint {id}? [y/N] ");
    if !confirm_yn(req.stdin_is_tty, req.answer.as_deref(), &question) {
        return RestoreOutcome::Skipped {
            reason: "user-project restore declined".into(),
        };
    }
    let flag_present = req
        .flag_present
        .unwrap_or_else(|| binary_supports_restore_flag(&req.hi_binary));
    if !flag_present {
        let kind = if id.starts_with("internal:v1:") {
            "internal:v1 snapshot"
        } else {
            "checkpoint"
        };
        eprintln!("hi: known-good hi lacks --sentinel-restore-checkpoint; skipping {kind} restore");
        return RestoreOutcome::Skipped {
            reason: "restore flag missing".into(),
        };
    }
    spawn_restore(&req.hi_binary, &req.workspace, id)
}

pub fn report_restore(outcome: &RestoreOutcome) {
    match outcome {
        RestoreOutcome::Restored { paths } => {
            eprintln!("hi: restored {paths} path(s)");
        }
        RestoreOutcome::Skipped { reason } => {
            eprintln!("hi: user-project restore skipped ({reason})");
        }
    }
}

fn spawn_restore(hi: &Path, workspace: &Path, id: &str) -> RestoreOutcome {
    let mut cmd = Command::new(hi);
    cmd.args([
        "--sentinel-restore-checkpoint",
        id,
        "--review-target",
        &workspace.display().to_string(),
    ])
    .env(ENV_ROLE, "restore")
    .env(ENV_SUPERVISED, "1")
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::inherit());
    match cmd.output() {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let paths = stdout
                .split_whitespace()
                .filter_map(|word| word.parse::<usize>().ok())
                .next()
                .unwrap_or(0);
            RestoreOutcome::Restored { paths }
        }
        Ok(output) => RestoreOutcome::Skipped {
            reason: format!("restore child exited {}", output.status),
        },
        Err(err) => RestoreOutcome::Skipped {
            reason: format!("restore spawn failed: {err}"),
        },
    }
}

pub fn binary_supports_restore_flag(hi: &Path) -> bool {
    let output = Command::new(hi)
        .args(["--sentinel-restore-checkpoint", "__hi_sentinel_probe__"])
        .env_remove(ENV_ROLE)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    let Ok(output) = output else {
        return false;
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    if text.contains("unexpected argument") || text.contains("unrecognized") {
        return false;
    }
    text.contains("HI_SENTINEL_ROLE=restore") || text.contains("sentinel-restore-checkpoint")
}

fn confirm_yn(stdin_is_tty: bool, injected: Option<&str>, question: &str) -> bool {
    if injected.is_none() && !stdin_is_tty {
        return false;
    }
    eprint!("{question}");
    let _ = io::stderr().flush();
    let line = match injected {
        Some(answer) => answer.to_string(),
        None => {
            let mut line = String::new();
            if io::stdin().read_line(&mut line).is_err() {
                return false;
            }
            line
        }
    };
    matches!(line.trim(), "y" | "Y" | "yes" | "YES")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsutil;

    fn stub(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fsutil::chmod_0700_file(&path).unwrap();
        path
    }

    #[test]
    fn apply_flag_does_not_skip_restore_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let hi = stub(dir.path(), "hi", "echo should-not-run >&2; exit 1");
        let outcome = maybe_restore_user_project(&RestoreRequest {
            hi_binary: hi,
            workspace: dir.path().to_path_buf(),
            pre_checkpoint: Some("abc123".into()),
            stdin_is_tty: false,
            answer: None,
            flag_present: Some(true),
        });
        assert_eq!(
            outcome,
            RestoreOutcome::Skipped {
                reason: "user-project restore declined".into()
            }
        );
    }

    #[test]
    fn internal_snapshot_skips_when_flag_missing() {
        let dir = tempfile::tempdir().unwrap();
        let hi = stub(
            dir.path(),
            "hi",
            "echo \"error: unexpected argument '--sentinel-restore-checkpoint'\" >&2; exit 2",
        );
        let outcome = maybe_restore_user_project(&RestoreRequest {
            hi_binary: hi,
            workspace: dir.path().to_path_buf(),
            pre_checkpoint: Some("internal:v1:deadbeef".into()),
            stdin_is_tty: true,
            answer: Some("y".into()),
            flag_present: None,
        });
        match outcome {
            RestoreOutcome::Skipped { reason } => {
                assert!(reason.contains("restore flag missing"), "{reason}");
            }
            other => panic!("expected skip, got {other:?}"),
        }
    }

    #[test]
    fn confirmed_restore_spawns_hidden_flag() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("args.log");
        let hi = stub(
            dir.path(),
            "hi",
            &format!(
                "printf '%s\\n' \"$HI_SENTINEL_ROLE\" \"$@\" > '{}'\necho 'restored 3 path(s)'\nexit 0",
                log.display()
            ),
        );
        let ws = dir.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let outcome = maybe_restore_user_project(&RestoreRequest {
            hi_binary: hi,
            workspace: ws.clone(),
            pre_checkpoint: Some("internal:v1:abc".into()),
            stdin_is_tty: true,
            answer: Some("y".into()),
            flag_present: Some(true),
        });
        assert_eq!(outcome, RestoreOutcome::Restored { paths: 3 });
        let logged = std::fs::read_to_string(&log).unwrap();
        assert!(logged.contains("restore"), "{logged}");
        assert!(logged.contains("--sentinel-restore-checkpoint"), "{logged}");
        assert!(logged.contains("internal:v1:abc"), "{logged}");
        assert!(logged.contains("--review-target"), "{logged}");
        assert!(
            logged.contains(&ws.display().to_string()),
            "restore must target the child workspace: {logged}"
        );
    }

    #[test]
    fn stub_without_flag_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let hi = stub(
            dir.path(),
            "old-hi",
            "echo \"error: unexpected argument '--sentinel-restore-checkpoint' found\" >&2; exit 2",
        );
        assert!(!binary_supports_restore_flag(&hi));
    }
}
