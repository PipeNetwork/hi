//! The recent-work scope of `/review`: what git says changed, with line
//! counts per file, grouped into turns when there is too much for one.
//!
//! Split from [`crate::review_scope`] (which decides *whether* to use it)
//! so both stay under the file-size ratchet.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::review_scope::{Unit, is_container, pack_units};

/// Commits scanned for churn ordering and for the "recent" file set.
pub const CHURN_COMMITS: usize = 300;

/// Untracked files longer than this are not line-counted (binaries, dumps).
const UNTRACKED_COUNT_LIMIT: u64 = 2 * 1024 * 1024;

/// Where a git-scoped file list came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GitSource {
    /// `git status`: modified, added, renamed, and untracked files.
    Uncommitted,
    /// The tree is clean; the files `HEAD` touched.
    LastCommit { short_hash: String, subject: String },
}

/// Lines added and removed in one file of the scope, per `git diff
/// --numstat`; an untracked file counts its lines as added.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitNumstat {
    pub path: String,
    /// `None` for a binary file.
    pub added: Option<u32>,
    pub removed: Option<u32>,
    #[serde(default)]
    pub untracked: bool,
}

impl GitNumstat {
    /// `+12 −3` / `untracked, 80 lines` / `binary`.
    pub fn label(&self) -> String {
        match (self.added, self.removed) {
            (Some(added), _) if self.untracked => format!("untracked, {added} lines"),
            (Some(added), Some(removed)) => format!("+{added} −{removed}"),
            _ => "binary".to_string(),
        }
    }
}

/// Recent work per git, chosen when no plan/spec and no scope were given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitScope {
    pub source: GitSource,
    /// Workspace-relative paths (`/` separators) that exist on disk, sorted.
    pub files: Vec<String>,
    /// Per-file line counts, for the files git could count.
    #[serde(default)]
    pub numstat: Vec<GitNumstat>,
    /// When the scope is audited in turns: the files of each turn, in the
    /// order of `ReviewInputs::chunks`. Empty for a single turn.
    #[serde(default)]
    pub groups: Vec<Vec<String>>,
}

impl GitScope {
    /// `47 uncommitted files (git status, +1240 −380)` / `last commit
    /// a1b2c3d (3 files, +12 −3)`.
    pub fn summary(&self) -> String {
        let count = self.files.len();
        let files = if count == 1 { "file" } else { "files" };
        let lines = self.line_totals();
        match &self.source {
            GitSource::Uncommitted => format!("{count} uncommitted {files} (git status{lines})"),
            GitSource::LastCommit { short_hash, .. } => {
                format!("last commit {short_hash} ({count} {files}{lines})")
            }
        }
    }

    /// `, +A −R` over the counted files; empty when nothing was counted.
    fn line_totals(&self) -> String {
        let (added, removed) = self.numstat.iter().fold((0u64, 0u64), |(a, r), stat| {
            (
                a + u64::from(stat.added.unwrap_or(0)),
                r + u64::from(stat.removed.unwrap_or(0)),
            )
        });
        if self.numstat.is_empty() {
            String::new()
        } else {
            format!(", +{added} −{removed}")
        }
    }

    pub fn numstat_for(&self, path: &str) -> Option<&GitNumstat> {
        self.numstat.iter().find(|stat| stat.path == path)
    }

    /// The files of chunk `index`, or every file when the scope is one turn.
    pub fn chunk_files(&self, index: usize) -> Vec<String> {
        if self.groups.is_empty() {
            return self.files.clone();
        }
        self.groups.get(index).cloned().unwrap_or_default()
    }

    /// Split the scope into turns of at most `budget` files, grouped by
    /// top-level directory (or by crate under a container such as
    /// `crates/`), small groups packed together. Returns the chunk labels
    /// for `ReviewInputs::chunks`; `groups` holds the files. A scope within
    /// the budget stays one turn and returns no labels.
    pub fn group(&mut self, root: &Path, budget: usize) -> Vec<String> {
        if self.files.len() <= budget {
            self.groups.clear();
            return Vec::new();
        }
        let mut units: Vec<Unit> = Vec::new();
        for file in &self.files {
            let key = group_key(root, file);
            match units.iter_mut().find(|unit| unit.label == key) {
                Some(unit) => unit.files.push(file.clone()),
                None => units.push(Unit {
                    label: key,
                    files: vec![file.clone()],
                    churn: 0,
                }),
            }
        }
        let packed = pack_units(units, budget);
        let (labels, groups) = packed
            .into_iter()
            .map(|unit| (unit.label, unit.files))
            .unzip();
        self.groups = groups;
        labels
    }
}

/// Grouping key of a scope file: its top-level directory, the crate under a
/// container directory, or `.` for a file in the workspace root.
fn group_key(root: &Path, file: &str) -> String {
    let mut parts = file.split('/');
    let (Some(first), Some(_rest)) = (parts.next(), parts.next()) else {
        return ".".to_string();
    };
    if is_container(&root.join(first)) {
        let mut parts = file.split('/');
        let first = parts.next().unwrap_or_default();
        match (parts.next(), parts.next()) {
            (Some(second), Some(_)) => format!("{first}/{second}"),
            _ => first.to_string(),
        }
    } else {
        first.to_string()
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
    let entries: Vec<(&str, &str)> = status_entries(&status).collect();
    let files = workspace_files(root, &repo_root, entries.iter().map(|(_, path)| *path));
    if !files.is_empty() {
        let mut numstat = parse_numstat(
            root,
            &repo_root,
            &git(
                root,
                &[
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "HEAD",
                    "--numstat",
                    "-z",
                    "--",
                    ".",
                ],
            )
            .unwrap_or_default(),
        );
        for (code, path) in &entries {
            if code.starts_with("??")
                && let Some(relative) = workspace_relative(root, &repo_root, path)
                && let Some(lines) = count_lines(&root.join(&relative))
            {
                numstat.push(GitNumstat {
                    path: relative,
                    added: Some(lines),
                    removed: Some(0),
                    untracked: true,
                });
            }
        }
        return Ok(GitScope {
            source: GitSource::Uncommitted,
            files,
            numstat,
            groups: Vec::new(),
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
            "--numstat",
            "-z",
            "HEAD",
            "--",
            ".",
        ],
    )
    .unwrap_or_default();
    let numstat = parse_numstat(root, &repo_root, &listed);
    let files = workspace_files(root, &repo_root, numstat_paths(&listed));
    if files.is_empty() {
        return Err(GitEmpty::Nothing);
    }
    Ok(GitScope {
        source: GitSource::LastCommit {
            short_hash: short_hash.to_string(),
            subject: subject.to_string(),
        },
        files,
        numstat,
        groups: Vec::new(),
    })
}

/// How many of the last [`CHURN_COMMITS`] commits touched each file under
/// `root`, by workspace-relative path (deleted files included). Empty
/// outside a repository.
pub fn churn_by_path(root: &Path) -> HashMap<String, usize> {
    let Some(toplevel) = git(root, &["rev-parse", "--show-toplevel"]) else {
        return HashMap::new();
    };
    let repo_root = PathBuf::from(toplevel.trim_end());
    let count = CHURN_COMMITS.to_string();
    let listed = git(
        root,
        &["log", "-n", &count, "--format=", "--name-only", "--", "."],
    )
    .unwrap_or_default();
    let root_abs = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut out: HashMap<String, usize> = HashMap::new();
    for path in listed
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let absolute = repo_root.join(path);
        let Ok(relative) = absolute.strip_prefix(&root_abs) else {
            continue;
        };
        let relative = relative.to_string_lossy().replace('\\', "/");
        *out.entry(relative).or_default() += 1;
    }
    out
}

/// Stdout of a successful `git` command run in `root`, else `None`. The
/// checkout's own `.git/config` must not get to run anything, so the
/// fsmonitor hook and untracked cache are off, as for the tools' git reads.
fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args([
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
        ])
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

/// `(XY, path)` from `git status --porcelain=v1 -z`: `XY path\0`, and for
/// a rename or copy `XY new\0old\0` (the old path is skipped).
fn status_entries(status: &str) -> impl Iterator<Item = (&str, &str)> {
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
            return Some((code, path));
        }
    })
}

/// Entries of `--numstat -z` output: `added\tremoved\tpath\0`, and for a
/// rename `added\tremoved\t\0old\0new\0` (the new path is kept). `-` counts
/// mark a binary file.
fn numstat_entries(listed: &str) -> Vec<(Option<u32>, Option<u32>, &str)> {
    let mut out = Vec::new();
    let mut fields = listed.split('\0');
    while let Some(entry) = fields.next() {
        let mut cells = entry.splitn(3, '\t');
        let (Some(added), Some(removed), Some(path)) = (cells.next(), cells.next(), cells.next())
        else {
            continue;
        };
        let path = if path.is_empty() {
            fields.next();
            match fields.next() {
                Some(new) => new,
                None => break,
            }
        } else {
            path
        };
        out.push((added.parse().ok(), removed.parse().ok(), path));
    }
    out
}

fn numstat_paths(listed: &str) -> impl Iterator<Item = &str> {
    numstat_entries(listed).into_iter().map(|(_, _, path)| path)
}

fn parse_numstat(root: &Path, repo_root: &Path, listed: &str) -> Vec<GitNumstat> {
    numstat_entries(listed)
        .into_iter()
        .filter_map(|(added, removed, path)| {
            Some(GitNumstat {
                path: workspace_relative(root, repo_root, path)?,
                added,
                removed,
                untracked: false,
            })
        })
        .collect()
}

/// Lines in a text file under the size limit; `None` for binaries and
/// large files.
fn count_lines(path: &Path) -> Option<u32> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > UNTRACKED_COUNT_LIMIT {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    if bytes.contains(&0) {
        return None;
    }
    let mut lines = bytes.iter().filter(|byte| **byte == b'\n').count();
    if !bytes.is_empty() && !bytes.ends_with(b"\n") {
        lines += 1;
    }
    u32::try_from(lines).ok()
}

/// A repository-relative `path` as a workspace-relative string, when it is
/// a file under `root`.
fn workspace_relative(root: &Path, repo_root: &Path, path: &str) -> Option<String> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let absolute = repo_root.join(path);
    let relative = absolute.strip_prefix(&root).ok()?;
    absolute
        .is_file()
        .then(|| relative.to_string_lossy().replace('\\', "/"))
}

/// Repository-relative `paths` as sorted, deduplicated workspace-relative
/// strings, keeping only files that exist.
fn workspace_files<'a>(
    root: &Path,
    repo_root: &Path,
    paths: impl Iterator<Item = &'a str>,
) -> Vec<String> {
    let mut files: Vec<String> = paths
        .filter_map(|path| workspace_relative(root, repo_root, path))
        .collect();
    files.sort();
    files.dedup();
    files
}

#[cfg(test)]
pub(crate) fn status_paths(status: &str) -> impl Iterator<Item = &str> {
    status_entries(status).map(|(_, path)| path)
}

#[cfg(test)]
pub(crate) fn numstat_entries_for_test(listed: &str) -> Vec<(Option<u32>, Option<u32>, String)> {
    numstat_entries(listed)
        .into_iter()
        .map(|(a, r, p)| (a, r, p.to_string()))
        .collect()
}

#[cfg(test)]
#[path = "review_git_tests.rs"]
pub(crate) mod tests;
