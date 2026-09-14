//! Strip Sentinel flags so the child cannot re-enter `maybe_exec`.

use std::ffi::OsString;
use std::path::PathBuf;

#[derive(Debug, Default)]
pub struct StrippedArgv {
    pub args: Vec<OsString>,
    pub apply: bool,
    pub checkout: Option<PathBuf>,
}

pub fn strip_sentinel_args(args: impl IntoIterator<Item = OsString>) -> StrippedArgv {
    let mut out = StrippedArgv::default();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        let Some(text) = arg.to_str() else {
            out.args.push(arg);
            continue;
        };
        match text {
            "--autoharnessfix" | "--no-autoharnessfix" => {}
            "--autoharnessfix-apply" | "--apply" => out.apply = true,
            "--autoharnessfix-checkout" | "--checkout" => {
                if let Some(value) = iter.next() {
                    out.checkout = Some(PathBuf::from(value));
                }
            }
            flag if flag.starts_with("--autoharnessfix-checkout=") => {
                out.checkout = Some(PathBuf::from(&flag["--autoharnessfix-checkout=".len()..]));
            }
            flag if flag.starts_with("--checkout=") => {
                out.checkout = Some(PathBuf::from(&flag["--checkout=".len()..]));
            }
            "--sentinel-restore-checkpoint" => {
                let _ = iter.next();
            }
            flag if flag.starts_with("--sentinel-restore-checkpoint=") => {}
            _ => out.args.push(arg),
        }
    }
    out
}

pub fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Drop the positional `Cli.prompt` operand. Relaunch is `--session-file` plus
/// original options only — forwarding the operand double-submits the turn.
pub fn drop_positional_prompt(args: impl IntoIterator<Item = OsString>) -> Vec<OsString> {
    let mut out = Vec::new();
    let mut iter = args.into_iter().peekable();
    while let Some(arg) = iter.next() {
        let Some(text) = arg.to_str().map(str::to_owned) else {
            out.push(arg);
            continue;
        };
        if text == "--" {
            break;
        }
        if let Some(rest) = text.strip_prefix("--") {
            let takes_value = !rest.contains('=') && !is_boolean_long(&text);
            out.push(arg);
            if !takes_value {
                continue;
            }
            let Some(next) = iter.peek() else {
                continue;
            };
            let next_text = next.to_str().unwrap_or("");
            if next_text == "--" || (next_text.starts_with('-') && next_text != "-") {
                continue;
            }
            out.push(iter.next().expect("peeked"));
            continue;
        }
        if text.starts_with('-') && text.len() == 2 && text != "-" {
            let boolean = is_boolean_short(&text);
            out.push(arg);
            if boolean {
                continue;
            }
            if let Some(next) = iter.next() {
                out.push(next);
            }
            continue;
        }
    }
    out
}

pub fn relaunch_args(
    child_args: &[OsString],
    session_path: Option<&std::path::Path>,
) -> Vec<OsString> {
    let mut args = drop_positional_prompt(child_args.iter().cloned());
    if let Some(path) = session_path
        && !has_flag(&args, "--session-file")
    {
        args.push(OsString::from("--session-file"));
        args.push(path.as_os_str().to_os_string());
    }
    args
}

fn has_flag(args: &[OsString], name: &str) -> bool {
    let prefix = format!("{name}=");
    args.iter().any(|arg| {
        arg.to_str()
            .is_some_and(|text| text == name || text.starts_with(&prefix))
    })
}

fn is_boolean_long(flag: &str) -> bool {
    matches!(
        flag,
        "--plain"
            | "--quiet"
            | "--yes"
            | "--continue"
            | "--no-save"
            | "--durable"
            | "--pipefs"
            | "--sync"
            | "--confirm-edits"
            | "--dry-run"
            | "--worktree"
            | "--no-verify"
            | "--allow-unverified"
            | "--keep-background"
            | "--no-auto-compact"
            | "--no-finalize"
            | "--no-memory"
            | "--skeptic-fail-open"
            | "--allow-no-checkpoint"
            | "--rsi"
            | "--no-rsi"
            | "--tasks"
            | "--no-tasks"
            | "--clippy"
            | "--rsi-managed"
            | "--benchmark-orchestration"
            | "--trace-full"
            | "--skeptic-review"
            | "--show-config"
            | "--list-sessions"
            | "--loops-daemon"
            | "--daemon"
            | "--resume-local"
            | "--subagent"
            | "--autoharnessfix"
            | "--no-autoharnessfix"
            | "--autoharnessfix-apply"
            | "--help"
            | "--version"
    )
}

fn is_boolean_short(flag: &str) -> bool {
    matches!(flag, "-q" | "-c" | "-h" | "-V" | "-y")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_parent_flags_and_keeps_prompt() {
        let stripped = strip_sentinel_args(
            [
                "--autoharnessfix",
                "--autoharnessfix-apply",
                "--autoharnessfix-checkout",
                "/tmp/hi",
                "--plain",
                "fix the parser",
            ]
            .into_iter()
            .map(OsString::from),
        );
        assert!(stripped.apply);
        assert_eq!(
            stripped.checkout.as_deref(),
            Some(std::path::Path::new("/tmp/hi"))
        );
        assert_eq!(
            stripped.args,
            vec![OsString::from("--plain"), OsString::from("fix the parser")]
        );
    }

    #[test]
    fn relaunch_drops_positional_prompt_from_original_argv() {
        let stripped = strip_sentinel_args(
            ["--autoharnessfix", "fix the parser"]
                .into_iter()
                .map(OsString::from),
        );
        let relaunch = relaunch_args(
            &stripped.args,
            Some(std::path::Path::new("/tmp/session.jsonl")),
        );
        assert_eq!(
            relaunch,
            vec![
                OsString::from("--session-file"),
                OsString::from("/tmp/session.jsonl"),
            ]
        );
        assert!(
            !relaunch.iter().any(|arg| arg == "fix the parser"),
            "positional prompt must not be forwarded: {relaunch:?}"
        );
    }

    #[test]
    fn relaunch_keeps_plain_and_drops_prompt() {
        let stripped = strip_sentinel_args(
            ["--plain", "fix the parser"]
                .into_iter()
                .map(OsString::from),
        );
        let relaunch = relaunch_args(&stripped.args, Some(std::path::Path::new("/s.jsonl")));
        assert_eq!(
            relaunch,
            vec![
                OsString::from("--plain"),
                OsString::from("--session-file"),
                OsString::from("/s.jsonl"),
            ]
        );
    }

    #[test]
    fn drop_positional_keeps_value_flags() {
        let args = drop_positional_prompt(
            [
                "--model",
                "pipe/x",
                "--review-target",
                "/tmp/proj",
                "fix the parser",
            ]
            .into_iter()
            .map(OsString::from),
        );
        assert_eq!(
            args,
            vec![
                OsString::from("--model"),
                OsString::from("pipe/x"),
                OsString::from("--review-target"),
                OsString::from("/tmp/proj"),
            ]
        );
    }
}
