//! Shared git-worktree helpers for isolated subprocess runs: best-of candidates
//! and the write-capable `delegate` subagent. A child `hi` works in a detached
//! worktree checked out to some base commit; on success only its verified diff
//! (relative to that base) is applied back to the real working tree.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Whether the current directory is inside a git work tree.
pub fn in_git_repo() -> bool {
    Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// A temp worktree path, namespaced by a caller `prefix` + `index`.
pub fn worktree_path(prefix: &str, index: u32) -> PathBuf {
    std::env::temp_dir().join(format!("hi-{prefix}-{}-{index}", std::process::id()))
}

/// Create a detached worktree at `path` checked out to `base` (a commit-ish).
pub fn add_worktree(path: &Path, base: &str) -> Result<()> {
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
pub fn verify_passes(worktree: &Path, verify: &str) -> bool {
    crate::prepare_verify_workdir(worktree);
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
pub fn apply_changes(worktree: &Path, base: &str) -> Result<bool> {
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
pub fn changed_files(worktree: &Path, base: &str) -> Vec<String> {
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
pub fn cleanup(worktrees: &[PathBuf]) {
    for path in worktrees {
        let _ = Command::new("git")
            .args(["worktree", "remove", "--force"])
            .arg(path)
            .output();
    }
}

/// Hard-reset a worktree onto a new base commit (fleet rebase: adopt a fresh
/// snapshot of the real tree, discarding the worktree's current state — callers
/// must ensure nothing unmerged is lost first).
pub fn reset_to(worktree: &Path, base: &str) -> Result<()> {
    let out = Command::new("git")
        .args(["reset", "--hard", base])
        .current_dir(worktree)
        .output()
        .context("running git reset in the worktree")?;
    if !out.status.success() {
        bail!(
            "git reset --hard {base} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end reset_to: a worktree with local edits hard-resets onto a new
    /// base commit, discarding its state and adopting the new snapshot.
    #[test]
    fn reset_to_adopts_a_new_base() {
        let dir = std::env::temp_dir().join(format!("hi-wt-reset-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let git = |args: &[&str], cwd: &Path| {
            let out = Command::new("git")
                .args(args)
                .current_dir(cwd)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        git(&["init", "-q"], &dir);
        std::fs::write(dir.join("a.txt"), "one\n").unwrap();
        git(&["add", "-A"], &dir);
        git(
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-qm",
                "c1",
            ],
            &dir,
        );
        let wt = dir.join("wt");
        git(
            &["worktree", "add", "-q", wt.to_str().unwrap(), "HEAD"],
            &dir,
        );
        // Dirty the worktree, then advance the base in the main repo.
        std::fs::write(wt.join("a.txt"), "dirty\n").unwrap();
        std::fs::write(dir.join("a.txt"), "two\n").unwrap();
        git(&["add", "-A"], &dir);
        git(
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-qm",
                "c2",
            ],
            &dir,
        );
        let new_base = git(&["rev-parse", "HEAD"], &dir);

        reset_to(&wt, &new_base).expect("reset succeeds");
        assert_eq!(std::fs::read_to_string(wt.join("a.txt")).unwrap(), "two\n");

        cleanup(&[wt]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
