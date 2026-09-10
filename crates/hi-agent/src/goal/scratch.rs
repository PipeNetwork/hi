//! Per-goal scratch directory for captured evidence (grok `{SCRATCH}`).
//!
//! Lives under the workspace `.hi/` tree so concurrent goals and skeptics do
//! not collide in shared `/tmp`. The harness never points `HOME` or toolchain
//! caches here — the directory is for throwaway output only.

use std::path::{Path, PathBuf};

/// Workspace-relative scratch path named in prompts.
pub(crate) const GOAL_SCRATCH_REL: &str = ".hi/scratch";

pub(crate) fn scratch_dir(root: &Path) -> PathBuf {
    root.join(GOAL_SCRATCH_REL)
}

/// Create the scratch directory. Best-effort: a failure must not block `/goal`.
pub(crate) fn ensure(root: &Path) -> Option<PathBuf> {
    let dir = scratch_dir(root);
    std::fs::create_dir_all(&dir).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }
    Some(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_creates_a_workspace_scratch_dir() {
        let root = std::env::temp_dir().join(format!(
            "hi-goal-scratch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let dir = ensure(&root).expect("scratch");
        assert!(dir.ends_with(GOAL_SCRATCH_REL));
        assert!(dir.is_dir());
        std::fs::remove_dir_all(root).unwrap();
    }
}
