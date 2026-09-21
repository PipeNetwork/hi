//! What code `/review` covers when the user named no scope.
//!
//! With a plan/spec, or on a small workspace, one audit turn covers the
//! whole tree. On a large workspace with nothing to compare against a
//! single turn cannot (a live run on this repo read six of 1175 files
//! against a README, spent 106k tokens, and reported "finding: none"), so
//! the audit narrows to recent work: the uncommitted changes, else the
//! files of the last commit. `all` audits the tree in chunks, one turn per
//! top-level directory (or per crate under a container such as `crates/`).
//! Only when git has nothing recent does the command refuse, saying what to
//! pass instead.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::review::{InputKind, ReviewInputs};

/// Workspaces with more non-ignored files than this are not audited whole
/// without a plan/spec: `/review` narrows to recent work instead.
pub const LARGE_WORKSPACE_FILES: usize = 200;

/// Chunks named in the start notice before `… (+N more)`.
const CHUNKS_LISTED: usize = 8;

/// Where a git-scoped file list came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GitSource {
    /// `git status`: modified, added, renamed, and untracked files.
    Uncommitted,
    /// The tree is clean; the files `HEAD` touched.
    LastCommit { short_hash: String, subject: String },
}

/// Recent work per git, chosen when no plan/spec and no scope were given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitScope {
    pub source: GitSource,
    /// Workspace-relative paths (`/` separators) that exist on disk, sorted.
    pub files: Vec<String>,
}

impl GitScope {
    /// `47 uncommitted files (git status)` / `last commit a1b2c3d (3 files)`.
    pub fn summary(&self) -> String {
        let count = self.files.len();
        let files = if count == 1 { "file" } else { "files" };
        match &self.source {
            GitSource::Uncommitted => format!("{count} uncommitted {files} (git status)"),
            GitSource::LastCommit { short_hash, .. } => {
                format!("last commit {short_hash} ({count} {files})")
            }
        }
    }
}

/// Why git offered no recent work to audit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GitEmpty {
    NotARepo,
    /// No uncommitted changes and no commit whose files are still in the
    /// tree (no commits at all, or an empty last commit).
    Nothing,
}

/// Decide what code the audit covers, given the discovered `inputs` and
/// whether `all` was asked for. Sets the git scope or the chunk list on
/// `inputs`, drops a README fallback when the audit is defects-only, and
/// writes the notice; `Err` is the user-facing refusal when there is
/// nothing to audit against and nothing recent to audit.
pub fn resolve_target(root: &Path, inputs: &mut ReviewInputs, all: bool) -> Result<(), String> {
    if all {
        let chunks = if inputs.scope.is_empty() {
            chunks(root)
        } else {
            std::mem::take(&mut inputs.scope)
        };
        if chunks.is_empty() {
            return Err("spec review: the workspace has no files to audit".into());
        }
        let what = if inputs.has_spec() {
            format!("auditing {} against the whole workspace", inputs.summary())
        } else {
            drop_readme(inputs);
            "no plan.md or spec.md found; auditing the whole workspace for defects".to_string()
        };
        inputs.notice = Some(format!(
            "{what} in {} chunks, one turn each: {}",
            chunks.len(),
            list_chunks(&chunks)
        ));
        inputs.chunks = chunks;
        return Ok(());
    }
    if inputs.has_spec()
        || !inputs.scope.is_empty()
        || !workspace_exceeds(root, LARGE_WORKSPACE_FILES)
    {
        return Ok(());
    }
    match git_scope(root) {
        Ok(scope) => {
            drop_readme(inputs);
            inputs.notice = Some(format!(
                "no plan.md or spec.md found; auditing recent work for defects: {} · `/review audit all` audits the whole repo in chunks",
                scope.summary()
            ));
            inputs.git = Some(scope);
            Ok(())
        }
        Err(empty) => Err(refusal(empty)),
    }
}

fn drop_readme(inputs: &mut ReviewInputs) {
    inputs.files.retain(|input| input.kind != InputKind::Readme);
}

fn list_chunks(chunks: &[String]) -> String {
    let mut text = chunks
        .iter()
        .take(CHUNKS_LISTED)
        .map(|chunk| chunk_label(chunk))
        .collect::<Vec<_>>()
        .join(", ");
    if chunks.len() > CHUNKS_LISTED {
        text.push_str(&format!(" (+{} more)", chunks.len() - CHUNKS_LISTED));
    }
    text
}

fn refusal(empty: GitEmpty) -> String {
    let why = match empty {
        GitEmpty::NotARepo => "this is not a git repository",
        GitEmpty::Nothing => {
            "git has no uncommitted changes and no commit whose files are still in the tree"
        }
    };
    format!(
        "spec review: no plan.md or spec.md found, the workspace has more than {LARGE_WORKSPACE_FILES} files, and {why}, \
so there is no recent work to audit. `/review audit all` audits the whole repo in chunks (one turn per top-level directory), \
`/review audit <dir>` one directory, or write plan.md (a `- [ ]` checklist of what should exist) for a coverage audit."
    )
}

/// `.` is the chunk for the workspace's own top-level files.
pub fn chunk_label(chunk: &str) -> String {
    if chunk == "." {
        "top-level files".to_string()
    } else {
        chunk.to_string()
    }
}

/// Recent work: the uncommitted changes (tracked and untracked, per
/// `git status`), else the files the last commit touched. Paths are
/// workspace-relative even when the workspace is a subdirectory of the
/// repository, and only files still on disk are kept.
pub fn git_scope(root: &Path) -> Result<GitScope, GitEmpty> {
    let toplevel = git(root, &["rev-parse", "--show-toplevel"]).ok_or(GitEmpty::NotARepo)?;
    let repo_root = PathBuf::from(toplevel.trim_end());
    // Porcelain paths are relative to the repository root whatever the cwd;
    // the `.` pathspec keeps them to this workspace.
    let status = git(
        root,
        &[
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            ".",
        ],
    )
    .unwrap_or_default();
    let files = workspace_files(root, &repo_root, status_paths(&status));
    if !files.is_empty() {
        return Ok(GitScope {
            source: GitSource::Uncommitted,
            files,
        });
    }
    let head = git(root, &["log", "-1", "--format=%h%x00%s"]).ok_or(GitEmpty::Nothing)?;
    let (short_hash, subject) = head
        .trim_end()
        .split_once('\0')
        .unwrap_or((head.trim(), ""));
    // `--root` diffs the first commit against the empty tree; `-m` lists a
    // merge's files against every parent instead of nothing.
    let listed = git(
        root,
        &[
            "diff-tree",
            "--root",
            "-r",
            "-m",
            "--no-commit-id",
            "--name-only",
            "-z",
            "HEAD",
            "--",
            ".",
        ],
    )
    .unwrap_or_default();
    let files = workspace_files(
        root,
        &repo_root,
        listed.split('\0').filter(|path| !path.is_empty()),
    );
    if files.is_empty() {
        return Err(GitEmpty::Nothing);
    }
    Ok(GitScope {
        source: GitSource::LastCommit {
            short_hash: short_hash.to_string(),
            subject: subject.to_string(),
        },
        files,
    })
}

/// Stdout of a successful `git` command run in `root`, else `None`.
fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Paths from `git status --porcelain=v1 -z`: `XY path\0`, and for a rename
/// or copy `XY new\0old\0` (the old path is skipped).
fn status_paths(status: &str) -> impl Iterator<Item = &str> {
    let mut fields = status.split('\0');
    std::iter::from_fn(move || {
        loop {
            let entry = fields.next()?;
            if !entry.is_char_boundary(3) || entry.len() < 4 {
                continue;
            }
            let (code, path) = entry.split_at(3);
            let renamed = code
                .bytes()
                .take(2)
                .any(|status| matches!(status, b'R' | b'C'));
            if renamed {
                fields.next();
            }
            return Some(path);
        }
    })
}

/// Repository-relative `paths` as sorted, deduplicated workspace-relative
/// strings, keeping only files that exist.
fn workspace_files<'a>(
    root: &Path,
    repo_root: &Path,
    paths: impl Iterator<Item = &'a str>,
) -> Vec<String> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut files: Vec<String> = paths
        .filter_map(|path| {
            let absolute = repo_root.join(path);
            let relative = absolute.strip_prefix(&root).ok()?;
            absolute
                .is_file()
                .then(|| relative.to_string_lossy().replace('\\', "/"))
        })
        .collect();
    files.sort();
    files.dedup();
    files
}

/// Directories audited one turn each for `all`: every non-ignored,
/// non-hidden top-level directory, except that a directory with no files of
/// its own (a container such as `crates/` or `packages/`) contributes its
/// subdirectories instead; the workspace's own top-level files are a last
/// chunk, `.`. Directories with no non-ignored files are skipped.
pub fn chunks(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let (dirs, root_has_files) = children(root);
    for dir in dirs {
        let (subdirs, has_files) = children(&root.join(&dir));
        if has_files || subdirs.is_empty() {
            if workspace_exceeds(&root.join(&dir), 0) {
                out.push(dir);
            }
            continue;
        }
        for sub in subdirs {
            let chunk = format!("{dir}/{sub}");
            if workspace_exceeds(&root.join(&chunk), 0) {
                out.push(chunk);
            }
        }
    }
    if root_has_files {
        out.push(".".into());
    }
    out
}

/// Sorted names of the non-ignored, non-hidden directories directly in
/// `dir`, and whether it holds any such file of its own.
fn children(dir: &Path) -> (Vec<String>, bool) {
    let mut dirs = Vec::new();
    let mut has_files = false;
    for entry in ignore::WalkBuilder::new(dir)
        .require_git(false)
        .max_depth(Some(1))
        .build()
        .flatten()
    {
        if entry.depth() == 0 {
            continue;
        }
        match entry.file_type() {
            Some(kind) if kind.is_dir() => {
                dirs.push(entry.file_name().to_string_lossy().into_owned());
            }
            Some(kind) if kind.is_file() => has_files = true,
            _ => {}
        }
    }
    dirs.sort();
    (dirs, has_files)
}

/// True when more than `limit` non-ignored, non-hidden files are under
/// `root`. The walk stops at `limit + 1`, so a huge tree costs the same as
/// a large one.
pub fn workspace_exceeds(root: &Path, limit: usize) -> bool {
    let mut count = 0usize;
    for entry in ignore::WalkBuilder::new(root)
        .require_git(false)
        .build()
        .flatten()
    {
        if entry.file_type().is_some_and(|kind| kind.is_file()) {
            count += 1;
            if count > limit {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
#[path = "review_scope_tests.rs"]
mod tests;
