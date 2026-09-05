//! Temporary indexes keep cancelled preparation from stranding real locks.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};
use tokio_util::sync::CancellationToken;

use super::command::{self, Budget};

pub(super) struct PrivateIndex {
    path: PathBuf,
    _directory: tempfile::TempDir,
}

impl PrivateIndex {
    pub(super) fn copy_from(
        worktree: &Path,
        budget: Budget,
        cancellation: Option<&CancellationToken>,
    ) -> Result<Self> {
        let mut locate = Command::new("git");
        locate.current_dir(worktree).args([
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "index",
        ]);
        let original = command::run(&mut locate, None, budget, cancellation)?;
        ensure!(original.status.success(), "could not locate worktree index");
        let original = std::str::from_utf8(&original.stdout)
            .context("non-UTF-8 index path")?
            .trim_end_matches('\n');
        let lock = PathBuf::from(format!("{original}.lock"));
        ensure!(
            !lock.exists(),
            "worktree index is locked by another Git operation"
        );
        let directory = tempfile::Builder::new()
            .prefix("hi-merge-index-")
            .tempdir()?;
        let index = Self {
            path: directory.path().join("index"),
            _directory: directory,
        };
        match std::fs::copy(original, &index.path) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let mut empty = Command::new("git");
                empty.current_dir(worktree).args(["read-tree", "--empty"]);
                index.configure(&mut empty);
                let empty = command::run(&mut empty, None, budget, cancellation)?;
                ensure!(empty.status.success(), "could not initialize private index");
            }
            Err(error) => return Err(error).context("copying worktree index"),
        }
        ensure!(
            !lock.exists(),
            "worktree index changed while preparing inspection"
        );
        Ok(index)
    }

    pub(super) fn configure(&self, command: &mut Command) {
        command
            .env("GIT_INDEX_FILE", &self.path)
            .env("GIT_OPTIONAL_LOCKS", "0");
    }
}
