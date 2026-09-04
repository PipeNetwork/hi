use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Give each firing its own path so a later attempt cannot clean up a
/// candidate retained after an inconclusive change inspection.
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
