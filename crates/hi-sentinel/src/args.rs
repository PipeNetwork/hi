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
}
