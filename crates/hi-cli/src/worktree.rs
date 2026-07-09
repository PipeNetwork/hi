//! Shared git-worktree helpers for isolated subprocess runs: best-of candidates
//! and the write-capable `delegate` subagent. A child `hi` works in a detached
//! worktree checked out to some base commit; on success only its verified diff
//! (relative to that base) is applied back to the real working tree.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Whether the current directory is inside a git work tree.
pub(crate) fn in_git_repo() -> bool {
    Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// A temp worktree path, namespaced by a caller `prefix` + `index`.
pub(crate) fn worktree_path(prefix: &str, index: u32) -> PathBuf {
    std::env::temp_dir().join(format!("hi-{prefix}-{}-{index}", std::process::id()))
}

/// Create a detached worktree at `path` checked out to `base` (a commit-ish).
pub(crate) fn add_worktree(path: &Path, base: &str) -> Result<()> {
    let output = Command::new("git")
        .args(["worktree", "add", "--detach"])
        .arg(path)
        .arg(base)
        .output()
        .context("running git worktree add")?;
    if !output.status.success() {
        bail!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Ground-truth check: run the verify command inside the worktree.
pub(crate) fn verify_passes(worktree: &Path, verify: &str) -> bool {
    hi_tools::prepare_verify_workdir(worktree);
    Command::new("sh")
        .arg("-c")
        .arg(verify)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .current_dir(worktree)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Apply the worktree's changes (relative to `base`, including new/deleted files)
/// to the main working tree. Returns `true` if any change was applied.
pub(crate) fn apply_changes(worktree: &Path, base: &str) -> Result<bool> {
    // Stage everything so the diff captures new/deleted files too.
    let add = Command::new("git")
        .current_dir(worktree)
        .args(["add", "-A"])
        .output()
        .context("git add in worktree")?;
    if !add.status.success() {
        bail!(
            "git add failed: {}",
            String::from_utf8_lossy(&add.stderr).trim()
        );
    }

    let diff = Command::new("git")
        .current_dir(worktree)
        .args(["diff", "--cached", base])
        .output()
        .context("git diff in worktree")?;
    if !diff.status.success() {
        bail!(
            "git diff failed: {}",
            String::from_utf8_lossy(&diff.stderr).trim()
        );
    }
    if diff.stdout.is_empty() {
        return Ok(false); // nothing changed
    }

    // Apply the patch in the main repo via stdin.
    use std::io::Write;
    let mut child = Command::new("git")
        .args(["apply", "--whitespace=nowarn"])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("spawning git apply")?;
    child
        .stdin
        .take()
        .context("git apply stdin")?
        .write_all(&diff.stdout)
        .context("writing patch to git apply")?;
    let status = child.wait().context("waiting for git apply")?;
    if !status.success() {
        bail!("git apply failed (working tree may conflict with the patch)");
    }
    Ok(true)
}

/// The files the worktree changed relative to `base` (staged first so new/deleted
/// files are included).
pub(crate) fn changed_files(worktree: &Path, base: &str) -> Vec<String> {
    let _ = Command::new("git")
        .current_dir(worktree)
        .args(["add", "-A"])
        .output();
    Command::new("git")
        .current_dir(worktree)
        .args(["diff", "--cached", "--name-only", base])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::to_string)
                .filter(|l| !l.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Force-remove the given worktrees.
pub(crate) fn cleanup(worktrees: &[PathBuf]) {
    for path in worktrees {
        let _ = Command::new("git")
            .args(["worktree", "remove", "--force"])
            .arg(path)
            .output();
    }
}
