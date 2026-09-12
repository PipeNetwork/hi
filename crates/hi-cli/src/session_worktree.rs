//! One-shot `--worktree` isolation using the CoW worktree helper.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hi_tools::worktree::{WorktreePlan, create_cow_worktree, remove_cow_worktree};

/// Isolated session worktree. Restores the previous cwd and deletes the
/// copy-on-write tree when dropped.
pub struct SessionWorktree {
    path: PathBuf,
    prev_cwd: PathBuf,
}

impl SessionWorktree {
    pub fn create(source: &Path) -> Result<Self> {
        let prev_cwd = std::env::current_dir().context("determining current directory")?;
        let path = std::env::temp_dir().join(format!(
            "hi-session-worktree-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        let plan = WorktreePlan::new(source, &path);
        create_cow_worktree(&plan).context("creating --worktree copy")?;
        std::env::set_current_dir(&path)
            .with_context(|| format!("changing into worktree {}", path.display()))?;
        Ok(Self { path, prev_cwd })
    }

    #[cfg(test)]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SessionWorktree {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.prev_cwd);
        let _ = remove_cow_worktree(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn git_ok(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?}");
    }

    #[test]
    fn creates_and_cleans_a_cow_worktree() {
        let dir = TempDir::new().unwrap();
        git_ok(dir.path(), &["init"]);
        git_ok(dir.path(), &["config", "user.email", "hi@example.com"]);
        git_ok(dir.path(), &["config", "user.name", "hi"]);
        std::fs::write(dir.path().join("README.md"), "hi\n").unwrap();
        git_ok(dir.path(), &["add", "README.md"]);
        git_ok(dir.path(), &["commit", "-qm", "init"]);
        let wt = SessionWorktree::create(dir.path()).unwrap();
        assert!(wt.path().join("README.md").is_file());
        let path = wt.path().to_path_buf();
        drop(wt);
        assert!(!path.exists());
    }
}
