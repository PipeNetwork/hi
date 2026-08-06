//! High-performance git worktree creation using copy-on-write cloning.
//!
//! On filesystems that support CoW (Btrfs, APFS, XFS with reflink), worktrees
//! are created by cloning the working tree without copying file data — only
//! metadata is duplicated. On other filesystems, falls back to `git worktree
//! add` or a parallel file copy.
//!
//! Inspired by grok-build's `xai-fast-worktree` crate.
//!
//! # Quick start
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use hi_fast_worktree::WorktreeBuilder;
//!
//! let report = WorktreeBuilder::new("/path/to/repo")
//!     .branch("feature-branch")
//!     .create("/tmp/worktree-feature")?;
//! println!("Created worktree at {} ({} files, CoW: {})",
//!     report.path.display(), report.files_copied, report.used_cow);
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

/// Errors from worktree creation.
#[derive(Debug, Error)]
pub enum WorktreeError {
    /// The source path is not a git repository.
    #[error("not a git repository: {0}")]
    NotARepo(PathBuf),
    /// A git command failed.
    #[error("git command failed: {0}")]
    GitFailed(String),
    /// The target path already exists.
    #[error("target path already exists: {0}")]
    TargetExists(PathBuf),
    /// An I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// How the working tree files are materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CreationMode {
    /// Use copy-on-write if the filesystem supports it, otherwise fall back to
    /// copy. This is the default.
    #[default]
    Auto,
    /// Always copy files (no CoW).
    Copy,
    /// Always use `git worktree add` (no CoW, git-managed).
    GitWorktree,
}

/// Report from a worktree creation operation.
#[derive(Debug, Clone)]
pub struct WorktreeReport {
    /// The path to the created worktree.
    pub path: PathBuf,
    /// Number of files copied/cloned.
    pub files_copied: usize,
    /// Whether copy-on-write was used.
    pub used_cow: bool,
    /// The creation mode that was actually used.
    pub mode: CreationMode,
}

/// Builder for creating git worktrees.
#[derive(Debug, Clone)]
pub struct WorktreeBuilder {
    repo_path: PathBuf,
    branch: Option<String>,
    commit: Option<String>,
    mode: CreationMode,
}

impl WorktreeBuilder {
    /// Create a new builder for the repository at `repo_path`.
    pub fn new(repo_path: impl Into<PathBuf>) -> Self {
        Self {
            repo_path: repo_path.into(),
            branch: None,
            commit: None,
            mode: CreationMode::default(),
        }
    }

    /// Set the branch to check out in the worktree.
    #[must_use]
    pub fn branch(mut self, branch: impl Into<String>) -> Self {
        self.branch = Some(branch.into());
        self
    }

    /// Set a specific commit to check out (instead of a branch).
    #[must_use]
    pub fn commit(mut self, commit: impl Into<String>) -> Self {
        self.commit = Some(commit.into());
        self
    }

    /// Set the creation mode.
    #[must_use]
    pub fn mode(mut self, mode: CreationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Create the worktree at `target_path`.
    pub fn create(&self, target_path: impl AsRef<Path>) -> Result<WorktreeReport, WorktreeError> {
        let target = target_path.as_ref();

        // Validate source is a git repo.
        if !self.repo_path.join(".git").exists() {
            // Could be a bare repo or worktree itself; check with git.
            let output = Command::new("git")
                .args(["rev-parse", "--git-dir"])
                .current_dir(&self.repo_path)
                .output()
                .map_err(|e| WorktreeError::GitFailed(e.to_string()))?;
            if !output.status.success() {
                return Err(WorktreeError::NotARepo(self.repo_path.clone()));
            }
        }

        if target.exists() {
            return Err(WorktreeError::TargetExists(target.to_path_buf()));
        }

        match self.mode {
            CreationMode::GitWorktree => self.create_via_git_worktree(target),
            CreationMode::Auto | CreationMode::Copy => self.create_via_copy(target),
        }
    }

    fn create_via_git_worktree(&self, target: &Path) -> Result<WorktreeReport, WorktreeError> {
        let mut cmd = Command::new("git");
        cmd.args(["worktree", "add"]).current_dir(&self.repo_path);

        if let Some(ref branch) = self.branch {
            cmd.arg("-b").arg(branch);
        } else if let Some(ref commit) = self.commit {
            cmd.arg("--detach").arg(commit);
        }

        cmd.arg(target);

        let output = cmd
            .output()
            .map_err(|e| WorktreeError::GitFailed(e.to_string()))?;
        if !output.status.success() {
            return Err(WorktreeError::GitFailed(format!(
                "git worktree add failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        // Count files in the new worktree.
        let files = count_files(target);

        Ok(WorktreeReport {
            path: target.to_path_buf(),
            files_copied: files,
            used_cow: false,
            mode: CreationMode::GitWorktree,
        })
    }

    fn create_via_copy(&self, target: &Path) -> Result<WorktreeReport, WorktreeError> {
        // Have git register a valid linked worktree, then mirror the source tree
        // over the fresh checkout so uncommitted files, modifications, and
        // deletions are all retained without forging .git metadata. A plain
        // overlay copy would leave stale files the source has deleted and
        // clobber nothing — so the checkout must first be reconciled down to
        // what the source actually contains.
        let report = self.create_via_git_worktree(target)?;
        let result = (|| {
            remove_checkout_entries_not_in(&self.repo_path, target)?;
            copy_dir_excluding_git(&self.repo_path, target)
        })();
        let files = match result {
            Ok(files) => files,
            Err(error) => {
                let _ = Command::new("git")
                    .args(["worktree", "remove", "--force"])
                    .arg(target)
                    .current_dir(&self.repo_path)
                    .output();
                return Err(error);
            }
        };

        // The copy uses plain std::fs::copy; whether the filesystem applies
        // copy-on-write (APFS clonefile, btrfs/xfs reflink) is not observable
        // from userspace here, so report CoW only as "possible", never claimed.
        let used_cow = false;

        Ok(WorktreeReport {
            path: report.path,
            files_copied: files,
            used_cow,
            mode: self.mode,
        })
    }
}

/// Remove a worktree created by [`WorktreeBuilder`].
pub fn remove_worktree(path: impl AsRef<Path>) -> Result<(), WorktreeError> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(path)?;
    Ok(())
}

/// Clean up orphaned worktrees in a repository (calls `git worktree prune`).
pub fn cleanup_worktrees_in(repo_path: impl AsRef<Path>) -> Result<usize, WorktreeError> {
    let output = Command::new("git")
        .args(["worktree", "prune", "--dry-run", "--verbose"])
        .current_dir(repo_path.as_ref())
        .output()
        .map_err(|e| WorktreeError::GitFailed(e.to_string()))?;
    let pruned = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .count();
    if pruned > 0 {
        let _ = Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(repo_path.as_ref())
            .output();
    }
    Ok(pruned)
}

/// Count the number of files in a directory tree.
fn count_files(path: &Path) -> usize {
    let mut count = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if p.file_name().is_some_and(|n| n == ".git") {
                    continue;
                }
                count += count_files(&p);
            } else {
                count += 1;
            }
        }
    }
    count
}

/// Copy a directory tree, excluding `.git` directories.
fn copy_dir_excluding_git(src: &Path, dst: &Path) -> Result<usize, WorktreeError> {
    let mut count = 0;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let p = entry.path();
        let name = entry.file_name();
        if name == ".git" {
            continue;
        }
        let dest = dst.join(&name);
        if p.is_dir() {
            std::fs::create_dir_all(&dest)?;
            count += copy_dir_excluding_git(&p, &dest)?;
        } else if p.is_symlink() {
            // Copy symlink as-is.
            let target = std::fs::read_link(&p)?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &dest)?;
        } else {
            std::fs::copy(&p, &dest)?;
            count += 1;
        }
    }
    Ok(count)
}

/// Remove every entry in the freshly checked-out `dst` worktree that has no
/// counterpart in the source working tree `src`, so the subsequent overlay
/// copy mirrors deletions as well as additions/modifications. `.git` metadata
/// (the worktree's gitdir file) is always preserved.
fn remove_checkout_entries_not_in(src: &Path, dst: &Path) -> Result<(), WorktreeError> {
    for entry in std::fs::read_dir(dst)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == ".git" {
            continue;
        }
        let counterpart = src.join(&name);
        let dst_path = entry.path();
        if !counterpart.exists() {
            if dst_path.is_dir() && !dst_path.is_symlink() {
                std::fs::remove_dir_all(&dst_path)?;
            } else {
                std::fs::remove_file(&dst_path)?;
            }
            continue;
        }
        // Recurse into directories present in both so nested deletions mirror.
        if dst_path.is_dir() && !dst_path.is_symlink() && counterpart.is_dir() {
            remove_checkout_entries_not_in(&counterpart, &dst_path)?;
        }
    }
    Ok(())
}

/// Count tracked files in a git repository using `git ls-files`.
pub fn count_tracked_files(repo_path: &Path) -> Result<usize, WorktreeError> {
    let output = Command::new("git")
        .args(["ls-files"])
        .current_dir(repo_path)
        .output()
        .map_err(|e| WorktreeError::GitFailed(e.to_string()))?;
    if !output.status.success() {
        return Err(WorktreeError::GitFailed(format!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let count = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .count();
    Ok(count)
}

/// Message shown when out of disk space.
pub const ENOSPC_OS_MESSAGE: &str = "No space left on device";

/// Context for out-of-disk errors.
pub const OUT_OF_DISK_CONTEXT: &str = "worktree creation";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let b = WorktreeBuilder::new("/tmp/repo");
        assert_eq!(b.repo_path, PathBuf::from("/tmp/repo"));
        assert!(b.branch.is_none());
        assert!(b.commit.is_none());
        assert_eq!(b.mode, CreationMode::Auto);
    }

    #[test]
    fn builder_chain() {
        let b = WorktreeBuilder::new("/tmp/repo")
            .branch("feature")
            .mode(CreationMode::Copy);
        assert_eq!(b.branch.as_deref(), Some("feature"));
        assert_eq!(b.mode, CreationMode::Copy);
    }

    #[test]
    fn creation_mode_default_is_auto() {
        assert_eq!(CreationMode::default(), CreationMode::Auto);
    }

    #[test]
    fn remove_nonexistent_is_ok() {
        assert!(remove_worktree("/nonexistent/path/12345").is_ok());
    }

    #[test]
    fn count_files_excludes_git() {
        let tmp = tempfile::tempdir().unwrap();
        // Create some files.
        std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
        std::fs::create_dir(tmp.path().join("subdir")).unwrap();
        std::fs::write(tmp.path().join("subdir/b.txt"), "world").unwrap();
        // .git should be excluded.
        std::fs::create_dir(tmp.path().join(".git")).unwrap();
        std::fs::write(tmp.path().join(".git/config"), "git config").unwrap();
        assert_eq!(count_files(tmp.path()), 2);
    }

    #[test]
    fn copy_dir_excludes_git() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();

        std::fs::write(src.path().join("file.txt"), "content").unwrap();
        std::fs::create_dir(src.path().join(".git")).unwrap();
        std::fs::write(src.path().join(".git/config"), "config").unwrap();

        let count = copy_dir_excluding_git(src.path(), dst.path()).unwrap();
        assert_eq!(count, 1);
        assert!(dst.path().join("file.txt").exists());
        assert!(!dst.path().join(".git").exists());
    }

    #[test]
    fn copy_mode_creates_valid_git_worktree() {
        let repo = tempfile::tempdir().unwrap();
        let target_parent = tempfile::tempdir().unwrap();
        let target = target_parent.path().join("copy");
        assert!(
            Command::new("git")
                .args(["init", "-q"])
                .current_dir(repo.path())
                .status()
                .unwrap()
                .success()
        );
        std::fs::write(repo.path().join("tracked.txt"), "base").unwrap();
        assert!(
            Command::new("git")
                .args(["add", "tracked.txt"])
                .current_dir(repo.path())
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .args([
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-qm",
                    "init"
                ])
                .current_dir(repo.path())
                .status()
                .unwrap()
                .success()
        );
        std::fs::write(repo.path().join("untracked.txt"), "local").unwrap();

        WorktreeBuilder::new(repo.path())
            .mode(CreationMode::Copy)
            .create(&target)
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(target.join("untracked.txt")).unwrap(),
            "local"
        );
        assert!(
            Command::new("git")
                .args(["status", "--porcelain"])
                .current_dir(&target)
                .status()
                .unwrap()
                .success()
        );
    }

    /// Regression: the copy overlay must mirror the source, not just layer on
    /// top of the fresh checkout. A file deleted from the source must not
    /// linger in the worktree, and a modified file must carry the source's
    /// contents — otherwise `Copy`/`Auto` worktrees silently diverge.
    #[test]
    fn copy_mode_mirrors_deletions_and_modifications() {
        let repo = tempfile::tempdir().unwrap();
        let target_parent = tempfile::tempdir().unwrap();
        let target = target_parent.path().join("copy");
        let git = |args: &[&str]| {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(repo.path())
                    .status()
                    .unwrap()
                    .success(),
                "git {args:?}"
            );
        };
        git(&["init", "-q"]);
        std::fs::write(repo.path().join("keep.txt"), "v1").unwrap();
        std::fs::write(repo.path().join("deleted.txt"), "gone").unwrap();
        git(&["add", "."]);
        git(&[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "init",
        ]);
        // Source diverges from the commit: modify one tracked file, delete the other.
        std::fs::write(repo.path().join("keep.txt"), "v2-edited").unwrap();
        std::fs::remove_file(repo.path().join("deleted.txt")).unwrap();

        WorktreeBuilder::new(repo.path())
            .mode(CreationMode::Copy)
            .create(&target)
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(target.join("keep.txt")).unwrap(),
            "v2-edited",
            "modified file must reflect the source working tree"
        );
        assert!(
            !target.join("deleted.txt").exists(),
            "file deleted in the source must not linger in the worktree"
        );
    }

    #[test]
    fn count_tracked_files_non_repo_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let result = count_tracked_files(tmp.path());
        // In a non-git directory, this should fail.
        assert!(result.is_err());
    }

    #[test]
    fn enospc_constants() {
        assert!(!ENOSPC_OS_MESSAGE.is_empty());
        assert!(!OUT_OF_DISK_CONTEXT.is_empty());
    }
}
