//! Detached git worktrees under the XDG cache dir. Sentinel commits; the agent does not.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::SystemTime;

use anyhow::{Context, Result, bail};

use crate::fsutil;

pub const MAX_WORKTREES: usize = 8;
pub const MAX_WORKTREE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub struct Worktree {
    pub path: PathBuf,
    pub base_sha: String,
}

pub struct FixCommit {
    pub branch: String,
    pub sha: String,
}

pub fn add_detached(checkout: &Path, dest: &Path, base: &str) -> Result<Worktree> {
    if let Some(parent) = dest.parent() {
        fsutil::mkdir_0700(parent)?;
    }
    if dest.exists() {
        let _ = fs::remove_dir_all(dest);
    }
    let output = git_in(
        checkout,
        &["worktree", "add", "--detach", dest_str(dest)?, base],
    )?;
    if !output.status.success() {
        bail!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let _ = fsutil::chmod_0700(dest);
    let base_sha = git_stdout(checkout, &["rev-parse", base])?;
    Ok(Worktree {
        path: dest.to_path_buf(),
        base_sha,
    })
}

pub fn remove(checkout: &Path, dest: &Path) {
    let dest_s = dest.to_string_lossy();
    let _ = git_in(
        checkout,
        &["worktree", "remove", "--force", dest_s.as_ref()],
    );
    let _ = fs::remove_dir_all(dest);
}

pub fn porcelain(worktree: &Path) -> Result<String> {
    git_stdout(worktree, &["status", "--porcelain=v1"])
}

pub fn has_changes(worktree: &Path) -> Result<bool> {
    Ok(!porcelain(worktree)?.is_empty())
}

#[cfg(test)]
pub fn current_branch(dir: &Path) -> Result<String> {
    git_stdout(dir, &["rev-parse", "--abbrev-ref", "HEAD"])
}

/// Create `autofix/<incident>` and commit all source changes. Never pushes.
pub fn commit_fix(worktree: &Path, incident_id: &str, kind: &str) -> Result<FixCommit> {
    if !has_changes(worktree)? {
        bail!("no fix produced");
    }
    let branch = format!("autofix/{incident_id}");
    let output = git_in(worktree, &["checkout", "-B", &branch])?;
    if !output.status.success() {
        bail!(
            "git checkout -B {branch} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let add = git_in(worktree, &["add", "-A"])?;
    if !add.status.success() {
        bail!(
            "git add failed: {}",
            String::from_utf8_lossy(&add.stderr).trim()
        );
    }
    if porcelain(worktree)?.is_empty() {
        bail!("no fix produced");
    }
    let message = format!("fix(sentinel): {incident_id} {kind}");
    let mut commit = Command::new("git");
    commit
        .arg("-C")
        .arg(worktree)
        .args([
            "-c",
            "user.name=hi-sentinel",
            "-c",
            "user.email=hi-sentinel@local",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--author=hi-sentinel <hi-sentinel@local>",
            "-m",
            &message,
        ])
        .env("GIT_AUTHOR_NAME", "hi-sentinel")
        .env("GIT_AUTHOR_EMAIL", "hi-sentinel@local")
        .env("GIT_COMMITTER_NAME", "hi-sentinel")
        .env("GIT_COMMITTER_EMAIL", "hi-sentinel@local")
        .env("GIT_TERMINAL_PROMPT", "0");
    let output = commit.output().context("git commit")?;
    if !output.status.success() {
        bail!(
            "git commit failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let sha = git_stdout(worktree, &["rev-parse", "HEAD"])?;
    Ok(FixCommit { branch, sha })
}

pub fn gc(parent: &Path, checkout: &Path, keep: Option<&Path>) {
    gc_with_caps(parent, checkout, keep, MAX_WORKTREES, MAX_WORKTREE_BYTES);
}

pub fn gc_with_caps(
    parent: &Path,
    checkout: &Path,
    keep: Option<&Path>,
    max_count: usize,
    max_bytes: u64,
) {
    let Ok(entries) = fs::read_dir(parent) else {
        return;
    };
    let mut dirs: Vec<(SystemTime, PathBuf, u64)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| keep.is_none_or(|k| k != p))
        .map(|p| {
            let modified = fs::metadata(&p)
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            let size = dir_size(&p);
            (modified, p, size)
        })
        .collect();
    dirs.sort_by_key(|(t, _, _)| *t);
    let mut total: u64 = dirs.iter().map(|(_, _, s)| *s).sum();
    let extra = dirs.len().saturating_sub(max_count);
    for (_, path, size) in dirs.iter().take(extra) {
        remove(checkout, path);
        total = total.saturating_sub(*size);
    }
    if total <= max_bytes {
        return;
    }
    let remain: Vec<PathBuf> = {
        let Ok(entries) = fs::read_dir(parent) else {
            return;
        };
        let mut v: Vec<(SystemTime, PathBuf, u64)> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .filter(|p| keep.is_none_or(|k| k != p))
            .map(|p| {
                let modified = fs::metadata(&p)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                let size = dir_size(&p);
                (modified, p, size)
            })
            .collect();
        v.sort_by_key(|(t, _, _)| *t);
        let mut bytes = v.iter().map(|(_, _, s)| *s).sum::<u64>();
        let mut drop = Vec::new();
        for (_, path, size) in v {
            if bytes <= max_bytes {
                break;
            }
            drop.push(path);
            bytes = bytes.saturating_sub(size);
        }
        drop
    };
    for path in remain {
        remove(checkout, &path);
    }
}

fn dest_str(path: &Path) -> Result<&str> {
    path.to_str().context("worktree path is not valid UTF-8")
}

fn git_in(dir: &Path, args: &[&str]) -> Result<Output> {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .with_context(|| format!("git {args:?}"))
}

fn git_stdout(dir: &Path, args: &[&str]) -> Result<String> {
    let output = git_in(dir, args)?;
    if !output.status.success() {
        bail!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn dir_size(path: &Path) -> u64 {
    fn walk(path: &Path) -> u64 {
        let Ok(meta) = fs::metadata(path) else {
            return 0;
        };
        if meta.is_file() {
            return meta.len();
        }
        let Ok(entries) = fs::read_dir(path) else {
            return 0;
        };
        entries.flatten().map(|e| walk(&e.path())).sum()
    }
    walk(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixture;

    #[test]
    fn add_remove_detached_does_not_move_checkout_branch() {
        let root = tempfile::tempdir().unwrap();
        let checkout = test_fixture::minimal_checkout(root.path());
        let before = current_branch(&checkout).unwrap();
        let dest = root.path().join("wt");
        let wt = add_detached(&checkout, &dest, "HEAD").unwrap();
        assert!(wt.path.join("crates/hi-cli/Cargo.toml").is_file());
        assert_eq!(current_branch(&checkout).unwrap(), before);
        remove(&checkout, &dest);
        assert!(!dest.exists());
        assert_eq!(current_branch(&checkout).unwrap(), before);
    }

    #[test]
    fn commit_lands_only_on_autofix_branch() {
        let root = tempfile::tempdir().unwrap();
        let checkout = test_fixture::minimal_checkout(root.path());
        let before = current_branch(&checkout).unwrap();
        let dest = root.path().join("wt");
        let wt = add_detached(&checkout, &dest, "HEAD").unwrap();
        fs::write(
            wt.path.join("crates/hi-harness/src/lib.rs"),
            "pub fn patched() {}\n",
        )
        .unwrap();
        let fix = commit_fix(&wt.path, "incident-1-aaaa", "invariant").unwrap();
        assert_eq!(fix.branch, "autofix/incident-1-aaaa");
        assert_eq!(current_branch(&wt.path).unwrap(), fix.branch);
        assert_eq!(current_branch(&checkout).unwrap(), before);
        assert_ne!(fix.sha, wt.base_sha);
        remove(&checkout, &dest);
    }

    #[test]
    fn gc_drops_oldest_over_count() {
        let root = tempfile::tempdir().unwrap();
        let checkout = test_fixture::minimal_checkout(root.path());
        let parent = root.path().join("wts");
        fsutil::mkdir_0700(&parent).unwrap();
        let keep = parent.join("keep");
        fsutil::mkdir_0700(&keep).unwrap();
        for i in 0..4 {
            fsutil::mkdir_0700(&parent.join(format!("old-{i}"))).unwrap();
        }
        gc_with_caps(&parent, &checkout, Some(&keep), 2, MAX_WORKTREE_BYTES);
        assert!(keep.is_dir());
        let remain: Vec<_> = fs::read_dir(&parent)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert!(remain.len() <= 3, "{remain:?}");
        assert!(remain.iter().any(|n| n == "keep"));
    }
}
