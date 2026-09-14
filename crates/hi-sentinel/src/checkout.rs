//! Checkout path compare. Guessing from `current_exe()` is forbidden.

use std::path::{Path, PathBuf};

pub fn canonical(path: &Path) -> Option<PathBuf> {
    std::fs::canonicalize(path).ok()
}

pub fn cwd_is_checkout(workspace: &str, checkout: Option<&Path>) -> bool {
    let Some(checkout) = checkout else {
        return false;
    };
    let Some(ws) = canonical(Path::new(workspace)) else {
        return false;
    };
    let Some(co) = canonical(checkout) else {
        return false;
    };
    ws == co
}

/// True when `path` looks like a Hi git checkout. Detection still runs if this fails.
pub fn validate(path: &Path) -> anyhow::Result<ValidatedCheckout> {
    anyhow::ensure!(path.is_dir(), "checkout is not a directory");
    let git = path.join(".git");
    anyhow::ensure!(git.exists(), "checkout is not a git repository");
    let cli_manifest = path.join("crates/hi-cli/Cargo.toml");
    anyhow::ensure!(cli_manifest.is_file(), "checkout is missing crates/hi-cli");
    let name = std::fs::read_to_string(&cli_manifest).unwrap_or_default();
    anyhow::ensure!(
        name.lines().any(|line| line.trim() == "name = \"hi\""),
        "crates/hi-cli is not package hi"
    );
    anyhow::ensure!(
        path.join("crates/hi-harness/Cargo.toml").is_file(),
        "checkout is missing crates/hi-harness"
    );
    let head = std::process::Command::new("git")
        .args(["-C", &path.display().to_string(), "rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let head_sha = head.ok_or_else(|| anyhow::anyhow!("could not read checkout HEAD"))?;
    let dirty = std::process::Command::new("git")
        .args([
            "-C",
            &path.display().to_string(),
            "status",
            "--porcelain=v1",
        ])
        .output()
        .ok()
        .is_some_and(|out| !out.stdout.is_empty());
    Ok(ValidatedCheckout {
        path: path.to_path_buf(),
        head_sha,
        dirty,
    })
}

#[derive(Clone, Debug)]
pub struct ValidatedCheckout {
    pub path: PathBuf,
    pub head_sha: String,
    pub dirty: bool,
}
