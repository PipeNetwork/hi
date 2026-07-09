//! Git-backed working-tree checkpoints for `/undo`.
//!
//! Before each turn the agent snapshots the full non-ignored working tree into a
//! *dangling* commit — built in a throwaway index so it never touches the user's
//! staging area, branch, or history. `/undo` restores the latest snapshot,
//! reverting every file the turn created, modified, or deleted in one step. This
//! is what makes running with no confirmation prompts safe: anything is undoable.
//!
//! Limitations: only works inside a git work tree; covers non-ignored files only
//! (build artifacts in `.gitignore` are left alone); file mode/symlink nuances
//! aren't preserved; and it can't undo non-file side effects (network, deletes
//! outside the repo) — those are what the catastrophic-op guard is for.

use std::path::Path;
use std::process::Output;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use tokio::process::Command;

async fn git(dir: &Path, args: &[&str]) -> Result<Output> {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .await
        .context("running git")
}

async fn git_indexed(dir: &Path, index: &str, args: &[&str]) -> Result<Output> {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_INDEX_FILE", index)
        .output()
        .await
        .context("running git")
}

async fn in_work_tree(dir: &Path) -> bool {
    git(dir, &["rev-parse", "--is-inside-work-tree"])
        .await
        .map(|o| o.status.success())
        .unwrap_or(false)
}

async fn toplevel(dir: &Path) -> Option<std::path::PathBuf> {
    let out = git(dir, &["rev-parse", "--show-toplevel"]).await.ok()?;
    if !out.status.success() {
        return None;
    }
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!p.is_empty()).then(|| std::path::PathBuf::from(p))
}

/// Snapshot the working tree of the repo containing `dir` into a dangling commit,
/// returning its SHA. `None` if `dir` isn't in a git work tree (so there's
/// nothing to checkpoint against).
pub async fn create(dir: &Path) -> Option<String> {
    if !in_work_tree(dir).await {
        return None;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!("hi-checkpoint-{}-{n}", std::process::id()));
    let index = tmp.to_str()?;

    // Seed the throwaway index from HEAD so `add -A` is a fast incremental
    // (harmlessly fails in a repo with no commits yet).
    let _ = git_indexed(dir, index, &["read-tree", "HEAD"]).await;
    let add = git_indexed(dir, index, &["add", "-A"]).await.ok()?;
    if !add.status.success() {
        let _ = std::fs::remove_file(&tmp);
        return None;
    }
    let tree_out = git_indexed(dir, index, &["write-tree"]).await.ok()?;
    let _ = std::fs::remove_file(&tmp);
    if !tree_out.status.success() {
        return None;
    }
    let tree = String::from_utf8_lossy(&tree_out.stdout).trim().to_string();

    let commit = git(dir, &["commit-tree", &tree, "-m", "hi checkpoint"])
        .await
        .ok()?;
    if !commit.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&commit.stdout).trim().to_string();
    (!sha.is_empty()).then_some(sha)
}

/// A unified diff of the working tree (of the repo containing `dir`) against
/// checkpoint `target` — everything that changed since that checkpoint, including
/// new and deleted files. Best-effort: `None` if not in a work tree, git errors,
/// or nothing changed. Used to show a reviewer what a turn actually did.
pub async fn diff(dir: &Path, target: &str) -> Option<String> {
    if !in_work_tree(dir).await {
        return None;
    }
    // Snapshot the current tree (captures untracked files too, via `add -A`) and
    // diff the checkpoint against it — the same technique `restore` uses, so new
    // files show up rather than being invisible to a bare `git diff <commit>`.
    let current = create(dir).await?;
    let out = git(dir, &["diff", "--no-renames", target, &current])
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let patch = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!patch.is_empty()).then_some(patch)
}

/// Restore the working tree to checkpoint `target`, undoing every change made
/// since. Returns the number of files restored or removed.
pub async fn restore(dir: &Path, target: &str) -> Result<usize> {
    let root = toplevel(dir).await.context("not in a git work tree")?;
    // Snapshot the current state and diff against the target, so we touch only
    // the files that actually changed (precise and safe).
    let current = create(dir)
        .await
        .context("couldn't snapshot current state")?;
    let diff = git(
        dir,
        &["diff", "--no-renames", "--name-status", target, &current],
    )
    .await?;
    if !diff.status.success() {
        bail!("git diff failed: {}", String::from_utf8_lossy(&diff.stderr));
    }

    let mut changed = 0usize;
    for line in String::from_utf8_lossy(&diff.stdout).lines() {
        let mut it = line.splitn(2, '\t');
        let status = it.next().unwrap_or("");
        let rel = it.next().unwrap_or("").trim();
        if rel.is_empty() {
            continue;
        }
        let abs = root.join(rel);
        match status.chars().next() {
            // Created since the checkpoint → remove it.
            Some('A') => {
                let _ = std::fs::remove_file(&abs);
                changed += 1;
            }
            // Modified/deleted/type-changed → rewrite the checkpoint's content.
            Some('M') | Some('D') | Some('T') => {
                let blob = git(dir, &["cat-file", "-p", &format!("{target}:{rel}")]).await?;
                if !blob.status.success() {
                    continue;
                }
                if let Some(parent) = abs.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                std::fs::write(&abs, &blob.stdout).with_context(|| format!("restoring {rel}"))?;
                changed += 1;
            }
            _ => {}
        }
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(dir: &Path, cmd: &str) {
        let ok = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(dir)
            .status()
            .unwrap()
            .success();
        assert!(ok, "command failed: {cmd}");
    }

    #[tokio::test]
    async fn checkpoint_restores_modified_created_and_deleted_files() {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "hi-ckpt-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        sh(
            &dir,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(dir.join("keep.txt"), "v1\n").unwrap();
        std::fs::write(dir.join("gone.txt"), "stays\n").unwrap();

        // Checkpoint the v1 state.
        let cp = create(&dir).await.expect("checkpoint");

        // A turn modifies one file, deletes another, and creates a third.
        std::fs::write(dir.join("keep.txt"), "v2 changed\n").unwrap();
        std::fs::remove_file(dir.join("gone.txt")).unwrap();
        std::fs::write(dir.join("new.txt"), "created by the turn\n").unwrap();

        let n = restore(&dir, &cp).await.expect("restore");
        assert_eq!(n, 3, "modified + deleted + created");
        assert_eq!(
            std::fs::read_to_string(dir.join("keep.txt")).unwrap(),
            "v1\n"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("gone.txt")).unwrap(),
            "stays\n"
        );
        assert!(!dir.join("new.txt").exists(), "created file removed");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
