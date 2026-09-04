use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, ensure};

use super::PYCACHE_EXCLUDES;

/// Stage and list the worktree's source changes. An inspection failure is not
/// evidence of an empty diff: callers must retain the candidate and report it.
pub fn changed_files(worktree: &Path, base: &str) -> Result<Vec<String>> {
    crate::prepare_verify_workdir(worktree);
    let staged = Command::new("git")
        .current_dir(worktree)
        .args(["add", "-A", "--", "."])
        .args(PYCACHE_EXCLUDES)
        .output()
        .context("staging worktree changes")?;
    ensure!(
        staged.status.success(),
        "staging worktree changes failed: {}",
        String::from_utf8_lossy(&staged.stderr).trim()
    );
    let diff = Command::new("git")
        .current_dir(worktree)
        .args([
            "diff",
            "--cached",
            "--no-renames",
            "--name-only",
            "-z",
            base,
            "--",
        ])
        .args(PYCACHE_EXCLUDES)
        .output()
        .context("inspecting worktree changes")?;
    ensure!(
        diff.status.success(),
        "inspecting worktree changes failed: {}",
        String::from_utf8_lossy(&diff.stderr).trim()
    );
    // -z avoids Git's display quoting of Unicode, tabs, quotes and newlines.
    // Renames are split so overlap checks see both the removed and added path.
    diff.stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            String::from_utf8(path.to_vec())
                .context("changed path is not valid UTF-8; cannot safely merge it")
        })
        .collect()
}

pub async fn changed_files_async(worktree: &Path, base: &str) -> Result<Vec<String>> {
    let worktree = worktree.to_path_buf();
    let base = base.to_owned();
    tokio::task::spawn_blocking(move || changed_files(&worktree, &base))
        .await
        .context("worktree change inspection worker failed")?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git(root: &Path, args: &[&str]) {
        let output = Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn repository() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        git(dir.path(), &["init", "-q"]);
        git(dir.path(), &["config", "user.name", "test"]);
        git(
            dir.path(),
            &["config", "user.email", "test@example.invalid"],
        );
        std::fs::write(dir.path().join("original.txt"), "original\n").unwrap();
        git(dir.path(), &["add", "-A"]);
        git(dir.path(), &["commit", "-qm", "base"]);
        dir
    }

    #[test]
    fn staging_failure_and_invalid_base_are_errors_not_empty_changes() {
        let dir = repository();
        std::fs::write(dir.path().join("candidate.rs"), "valuable new source\n").unwrap();
        std::fs::write(dir.path().join(".git/index.lock"), "busy").unwrap();
        let error = changed_files(dir.path(), "HEAD").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("staging worktree changes failed")
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("candidate.rs")).unwrap(),
            "valuable new source\n"
        );
        std::fs::remove_file(dir.path().join(".git/index.lock")).unwrap();
        assert!(
            changed_files(dir.path(), "missing-baseline")
                .unwrap_err()
                .to_string()
                .contains("inspecting worktree changes failed")
        );
        assert_eq!(changed_files(dir.path(), "HEAD").unwrap(), ["candidate.rs"]);
    }

    #[test]
    fn rename_lists_both_paths_and_preserves_display_quoted_names() {
        let dir = repository();
        let renamed = "renamed\n\"日本語\".txt";
        std::fs::rename(dir.path().join("original.txt"), dir.path().join(renamed)).unwrap();
        let changed = changed_files(dir.path(), "HEAD").unwrap();
        assert_eq!(changed.len(), 2, "{changed:?}");
        assert!(changed.contains(&"original.txt".to_owned()));
        assert!(changed.contains(&renamed.to_owned()), "{changed:?}");
    }

    #[tokio::test]
    async fn asynchronous_inspection_preserves_errors_and_legitimate_empty_diffs() {
        let dir = repository();
        assert!(
            changed_files_async(dir.path(), "HEAD")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            changed_files_async(dir.path(), "missing-baseline")
                .await
                .is_err()
        );
    }
}
