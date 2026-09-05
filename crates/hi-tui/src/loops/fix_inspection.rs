use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Give each firing its own path so a later attempt cannot clean up a
/// candidate retained after inspection, verification, or publication fails.
pub(super) fn worktree_path(loop_id: u64, base: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "hi-loopfix-{}-{loop_id}-{}-{}-{}",
        std::process::id(),
        base.chars().take(12).collect::<String>(),
        super::now_ms(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ))
}

pub(super) async fn inspect(worktree: &Path, base: &str) -> Result<Vec<String>, (String, bool)> {
    hi_tools::worktree::changed_files_async(worktree, base)
        .await
        .map_err(|error| {
            (
                format!(
                    "could not inspect fix changes: {error:#}; candidate retained at {}",
                    worktree.display()
                ),
                true,
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn inspection_failure_preserves_candidate_and_next_firing_uses_a_new_path() {
        let candidate = tempfile::tempdir().unwrap();
        std::fs::write(candidate.path().join("source.rs"), "new work").unwrap();
        let (message, loud) = inspect(candidate.path(), "missing-base").await.unwrap_err();
        assert!(loud);
        assert!(message.contains(&candidate.path().display().to_string()));
        assert!(message.contains("candidate retained"));
        assert_eq!(
            std::fs::read_to_string(candidate.path().join("source.rs")).unwrap(),
            "new work"
        );
        assert_ne!(worktree_path(1, "same-base"), worktree_path(1, "same-base"));
    }
}

#[cfg(all(test, unix))]
mod recovery_tests {
    use super::super::{run_fix, tests};
    use tokio_util::sync::CancellationToken;

    fn retained_path(message: &str) -> std::path::PathBuf {
        message
            .rsplit_once("; candidate retained at ")
            .unwrap_or_else(|| panic!("missing recovery path: {message}"))
            .1
            .into()
    }

    #[tokio::test]
    async fn rejected_fix_retains_the_candidate_for_recovery() {
        let root = tempfile::tempdir().unwrap();
        tests::init_git_repo(root.path());
        let launcher = tests::fix_launcher(
            root.path(),
            tests::fixer_stub(root.path(), "retain-rejected.sh", "candidate.rs"),
            Some("false"),
        );
        let result = run_fix(
            &launcher,
            &tests::spec(),
            "fix it",
            CancellationToken::new(),
        )
        .await;
        assert!(result.0.contains("NOT merged"), "{result:?}");
        assert!(!root.path().join("candidate.rs").exists());
        let candidate = retained_path(&result.0);
        assert_eq!(
            std::fs::read_to_string(candidate.join("candidate.rs")).unwrap(),
            "patched"
        );
        super::super::cleanup_loop_fix(root.path(), &candidate).await;
    }

    #[tokio::test]
    async fn merge_conflict_preserves_both_user_changes_and_candidate() {
        let root = tempfile::tempdir().unwrap();
        tests::init_git_repo(root.path());
        let exe = tests::fixer_stub(root.path(), "retain-conflict.sh", "README");
        let user_path = root.path().join("README");
        let quoted_path = format!(
            "'{}'",
            user_path.display().to_string().replace('\'', "'\\''")
        );
        // The user's change arrives after the candidate's base was captured.
        std::fs::write(
            &exe,
            format!("#!/bin/sh\nprintf 'candidate' > README\nprintf 'user edit' > {quoted_path}\n"),
        )
        .unwrap();
        let launcher = tests::fix_launcher(root.path(), exe, Some("true"));
        let result = run_fix(
            &launcher,
            &tests::spec(),
            "fix it",
            CancellationToken::new(),
        )
        .await;
        assert!(result.0.contains("merge failed"), "{result:?}");
        assert_eq!(std::fs::read_to_string(user_path).unwrap(), "user edit");
        let candidate = retained_path(&result.0);
        assert_eq!(
            std::fs::read_to_string(candidate.join("README")).unwrap(),
            "candidate"
        );
        super::super::cleanup_loop_fix(root.path(), &candidate).await;
    }

    #[tokio::test]
    async fn failed_pr_commit_keeps_the_uncommitted_fix() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        tests::init_git_repo(root.path());
        let hook = root.path().join(".git/hooks/pre-commit");
        std::fs::write(&hook, "#!/bin/sh\nexit 1\n").unwrap();
        std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
        let launcher = tests::fix_launcher(
            root.path(),
            tests::fixer_stub(root.path(), "retain-pr-commit.sh", "candidate.rs"),
            Some("true"),
        );
        let mut spec = tests::spec();
        spec.fix_pr = true;
        let result = run_fix(&launcher, &spec, "fix it", CancellationToken::new()).await;
        assert!(
            result.0.contains("couldn't prepare the PR branch"),
            "{result:?}"
        );
        assert!(!root.path().join("candidate.rs").exists());
        let candidate = retained_path(&result.0);
        assert_eq!(
            std::fs::read_to_string(candidate.join("candidate.rs")).unwrap(),
            "patched"
        );
        super::super::cleanup_loop_fix(root.path(), &candidate).await;
    }

    #[tokio::test]
    async fn successful_pr_commit_preserves_uncommitted_hook_changes() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        tests::init_git_repo(root.path());
        let hook = root.path().join(".git/hooks/post-commit");
        std::fs::write(&hook, "#!/bin/sh\nprintf 'hook changes' > candidate.rs\n").unwrap();
        std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
        let launcher = tests::fix_launcher(
            root.path(),
            tests::fixer_stub(root.path(), "retain-pr-hook.sh", "candidate.rs"),
            Some("true"),
        );
        let mut spec = tests::spec();
        spec.fix_pr = true;
        // No remote is configured, so publication stops after the local commit.
        let result = run_fix(&launcher, &spec, "fix it", CancellationToken::new()).await;
        assert!(result.0.contains("fix committed to branch"), "{result:?}");
        assert!(result.0.contains("additional work remains"), "{result:?}");
        let candidate = retained_path(&result.0);
        assert_eq!(
            std::fs::read_to_string(candidate.join("candidate.rs")).unwrap(),
            "hook changes"
        );
        let committed = std::process::Command::new("git")
            .current_dir(&candidate)
            .args(["show", "HEAD:candidate.rs"])
            .output()
            .unwrap();
        assert!(committed.status.success());
        assert_eq!(committed.stdout, b"patched");
        super::super::cleanup_loop_fix(root.path(), &candidate).await;
    }
}
