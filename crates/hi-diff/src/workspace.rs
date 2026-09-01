//! Read-only source capture and isolated worktree helpers for agent runs.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

use crate::WorkspaceSnapshot;

pub fn capture_workspace_snapshot(root: &Path) -> Result<WorkspaceSnapshot> {
    let revision = git(root, &["rev-parse", "HEAD"]).ok();
    let dirty = git(root, &["diff", "HEAD", "--no-ext-diff", "--binary"])?
        .trim()
        .to_string();
    Ok(WorkspaceSnapshot {
        root: root.to_path_buf(),
        source_revision: revision.filter(|value| !value.is_empty()),
        dirty_patch: (!dirty.is_empty()).then_some(dirty),
    })
}

/// Materialize one independent agent worktree from a captured source state.
/// The caller owns cleanup through [`remove_isolated_worktree`].
pub fn create_isolated_worktree(
    snapshot: &WorkspaceSnapshot,
    destination: &Path,
) -> Result<PathBuf> {
    let revision = snapshot
        .source_revision
        .as_deref()
        .context("isolated worktrees require a git HEAD revision")?;
    if destination.exists() {
        bail!(
            "isolated worktree destination already exists: {}",
            destination.display()
        );
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    run_git(
        &snapshot.root,
        &[
            "worktree",
            "add",
            "--detach",
            &destination.to_string_lossy(),
            revision,
        ],
        None,
    )?;
    if let Some(patch) = &snapshot.dirty_patch {
        run_git(
            destination,
            &["apply", "--whitespace=nowarn", "--binary", "-"],
            Some(patch.as_bytes()),
        )?;
    }
    Ok(destination.to_path_buf())
}

pub fn remove_isolated_worktree(snapshot: &WorkspaceSnapshot, destination: &Path) -> Result<()> {
    run_git(
        &snapshot.root,
        &[
            "worktree",
            "remove",
            "--force",
            &destination.to_string_lossy(),
        ],
        None,
    )
}

fn git(root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .trim_end()
        .to_string())
}

fn run_git(root: &Path, args: &[&str], stdin: Option<&[u8]>) -> Result<()> {
    let mut command = Command::new("git");
    command.arg("-C").arg(root).args(args);
    if stdin.is_some() {
        command.stdin(Stdio::piped());
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("starting git {}", args.join(" ")))?;
    if let Some(input) = stdin
        && let Some(mut pipe) = child.stdin.take()
    {
        pipe.write_all(input)?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_reports_revision_and_dirty_patch() {
        let temp = tempfile::tempdir().unwrap();
        run_git(temp.path(), &["init", "-q"], None).unwrap();
        std::fs::write(temp.path().join("a.txt"), "one\n").unwrap();
        run_git(temp.path(), &["add", "a.txt"], None).unwrap();
        run_git(
            temp.path(),
            &[
                "-c",
                "user.name=Diff Lab",
                "-c",
                "user.email=diff@example.invalid",
                "commit",
                "-qm",
                "seed",
            ],
            None,
        )
        .unwrap();
        std::fs::write(temp.path().join("a.txt"), "two\n").unwrap();
        let snapshot = capture_workspace_snapshot(temp.path()).unwrap();
        assert!(snapshot.source_revision.is_some());
        assert!(snapshot.dirty_patch.is_some());
    }
}
