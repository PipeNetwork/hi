//! Deterministic diff-hygiene checks after a green WorkspaceRepair pass.
//!
//! These are conservative merge-quality signals, not compile/test authority.
//! Findings re-enter the model like an independent-review OBJECT.

use std::collections::BTreeSet;
use std::fs::Metadata;
use std::io::Read;
use std::path::{Component, Path};
use std::process::Stdio;
use std::time::Duration;

use hi_tools::{FileChange, FileChangeKind};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::task_contract::TaskContract;

/// Cap (bytes) above which a single Create/Modify is a hygiene finding.
pub(crate) const LARGE_FILE_BYTES: u64 = 32 * 1024;
/// Unreferenced Creates that trip the sprawl check on a narrow mutation.
const UNREFERENCED_CREATE_THRESHOLD: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HygieneFinding {
    pub reason: String,
}

/// Assess only changes that could appear in a repository diff.
///
/// Workspace effect accounting deliberately includes ignored files because
/// configuration and runtime data still matter for durability, cancellation,
/// and recovery. Diff hygiene is narrower: an ignored, untracked runtime file
/// cannot be merged, so treating a database reset as a large source rewrite
/// creates a permanent repair loop. Git's default `check-ignore` behavior does
/// not report tracked paths, even when a later ignore rule matches them, which
/// preserves hygiene coverage for tracked source.
///
/// Non-Git workspaces and Git failures fail conservatively by retaining every
/// change.
pub(crate) async fn assess_reviewable(
    root: &Path,
    contract: &TaskContract,
    changes: &[FileChange],
    prompt: &str,
) -> Vec<HygieneFinding> {
    let reviewable = reviewable_changes(root, changes).await;
    assess(contract, &reviewable, prompt)
}

async fn reviewable_changes(root: &Path, changes: &[FileChange]) -> Vec<FileChange> {
    let Some(ignored) = ignored_untracked_paths(root, changes).await else {
        return changes.to_vec();
    };
    changes
        .iter()
        .filter(|change| !ignored.contains(change.path.as_bytes()))
        .cloned()
        .collect()
}

/// Exact content identity for the cumulative, reviewable paths changed by the
/// turn. Unlike the workspace ledger's cheap whole-tree revision, this reads
/// oversized files in full so an equal-length rewrite cannot masquerade as no
/// progress. Ignored untracked runtime paths are intentionally absent because
/// neither diff hygiene nor completion review can act on them.
///
/// `None` means the filesystem could not be observed consistently. Callers
/// must treat that as inconclusive and allow the repair, never as equality.
pub(crate) async fn reviewable_content_revision(
    root: &Path,
    changes: &[FileChange],
    cancellation: Option<crate::TurnCancellation>,
) -> Option<String> {
    let mut paths = reviewable_changes(root, changes)
        .await
        .into_iter()
        .map(|change| change.path)
        .collect::<Vec<_>>();
    paths.sort();
    paths.dedup();

    let root = root.to_path_buf();
    tokio::task::spawn_blocking(move || {
        reviewable_content_revision_blocking(&root, &paths, cancellation.as_ref())
    })
    .await
    .ok()
    .flatten()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReviewableNodeKind {
    File,
    Symlink,
    Directory,
    Other,
}

impl ReviewableNodeKind {
    fn label(self) -> &'static [u8] {
        match self {
            Self::File => b"file",
            Self::Symlink => b"symlink",
            Self::Directory => b"directory",
            Self::Other => b"other",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReviewableMetadata {
    kind: ReviewableNodeKind,
    mode: u32,
    len: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    modified: (i64, i64),
    #[cfg(unix)]
    changed: (i64, i64),
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
}

impl ReviewableMetadata {
    fn from(metadata: &Metadata) -> Self {
        let file_type = metadata.file_type();
        let kind = if file_type.is_file() {
            ReviewableNodeKind::File
        } else if file_type.is_symlink() {
            ReviewableNodeKind::Symlink
        } else if file_type.is_dir() {
            ReviewableNodeKind::Directory
        } else {
            ReviewableNodeKind::Other
        };
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            Self {
                kind,
                mode: reviewable_file_mode(metadata),
                len: metadata.len(),
                device: metadata.dev(),
                inode: metadata.ino(),
                modified: (metadata.mtime(), metadata.mtime_nsec()),
                changed: (metadata.ctime(), metadata.ctime_nsec()),
            }
        }
        #[cfg(not(unix))]
        {
            Self {
                kind,
                mode: reviewable_file_mode(metadata),
                len: metadata.len(),
                modified: metadata.modified().ok(),
            }
        }
    }
}

fn reviewable_content_revision_blocking(
    root: &Path,
    paths: &[String],
    cancellation: Option<&crate::TurnCancellation>,
) -> Option<String> {
    let root = root.canonicalize().ok()?;
    let mut hash = Sha256::new();
    hash.update(b"hi-agent:reviewable-content-revision:v1\0");
    hash.update(u64::try_from(paths.len()).ok()?.to_be_bytes());
    for relative in paths {
        if hashing_cancelled(cancellation) {
            return None;
        }
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return None;
        }
        let path = contained_reviewable_path(&root, relative_path)?;
        let parent = path.parent()?.to_path_buf();
        hash_length_prefixed(&mut hash, relative.as_bytes())?;
        hash_reviewable_node(&mut hash, &path, cancellation)?;
        // Re-resolve the parent after reading. A directory or intermediate
        // symlink race makes the result inconclusive rather than binding an
        // equality decision to bytes reached outside this workspace path.
        if parent.canonicalize().ok()? != parent || !parent.starts_with(&root) {
            return None;
        }
    }
    Some(format!(
        "reviewable-content:v1:sha256:{:x}",
        hash.finalize()
    ))
}

fn contained_reviewable_path(root: &Path, relative: &Path) -> Option<std::path::PathBuf> {
    let file_name = relative.file_name()?;
    let parent = root.join(relative).parent()?.canonicalize().ok()?;
    if !parent.starts_with(root) {
        return None;
    }
    Some(parent.join(file_name))
}

fn hash_reviewable_node(
    hash: &mut Sha256,
    path: &Path,
    cancellation: Option<&crate::TurnCancellation>,
) -> Option<()> {
    if hashing_cancelled(cancellation) {
        return None;
    }
    let before = match std::fs::symlink_metadata(path) {
        Ok(metadata) => ReviewableMetadata::from(&metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // A second lookup distinguishes a stable deletion from a node
            // appearing during the observation window.
            match std::fs::symlink_metadata(path) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    hash_length_prefixed(hash, b"missing")?;
                    hash_length_prefixed(hash, &[])?;
                    hash_length_prefixed(hash, &[])?;
                    hash_length_prefixed(hash, &[])?;
                    return Some(());
                }
                _ => return None,
            }
        }
        Err(_) => return None,
    };

    hash_length_prefixed(hash, before.kind.label())?;
    hash_length_prefixed(hash, &before.mode.to_be_bytes())?;
    hash_length_prefixed(hash, &before.len.to_be_bytes())?;

    match before.kind {
        ReviewableNodeKind::File => {
            if hashing_cancelled(cancellation) {
                return None;
            }
            let mut options = std::fs::OpenOptions::new();
            options.read(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt as _;
                // Refuse a symlink swap and avoid blocking if a path is raced
                // to a FIFO between lstat and open.
                options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK);
            }
            let mut file = options.open(path).ok()?;
            if ReviewableMetadata::from(&file.metadata().ok()?) != before {
                return None;
            }
            hash.update(before.len.to_be_bytes());
            let mut observed_len = 0_u64;
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                if hashing_cancelled(cancellation) {
                    return None;
                }
                let read = file.read(&mut buffer).ok()?;
                if hashing_cancelled(cancellation) {
                    return None;
                }
                if read == 0 {
                    break;
                }
                observed_len = observed_len.checked_add(u64::try_from(read).ok()?)?;
                if observed_len > before.len {
                    return None;
                }
                hash.update(&buffer[..read]);
            }
            if observed_len != before.len
                || ReviewableMetadata::from(&file.metadata().ok()?) != before
                || ReviewableMetadata::from(&std::fs::symlink_metadata(path).ok()?) != before
            {
                return None;
            }
        }
        ReviewableNodeKind::Symlink => {
            if hashing_cancelled(cancellation) {
                return None;
            }
            let target = std::fs::read_link(path).ok()?;
            if hashing_cancelled(cancellation) {
                return None;
            }
            let bytes = target.as_os_str().as_encoded_bytes();
            hash_length_prefixed(hash, bytes)?;
            if ReviewableMetadata::from(&std::fs::symlink_metadata(path).ok()?) != before {
                return None;
            }
        }
        // File changes should resolve only to regular files, links, or stable
        // absence. Metadata alone is not exact content evidence for a
        // directory, device, socket, or FIFO.
        ReviewableNodeKind::Directory | ReviewableNodeKind::Other => return None,
    }
    Some(())
}

fn hashing_cancelled(cancellation: Option<&crate::TurnCancellation>) -> bool {
    cancellation.is_some_and(crate::TurnCancellation::is_cancelled)
}

fn hash_length_prefixed(hash: &mut Sha256, bytes: &[u8]) -> Option<()> {
    hash.update(u64::try_from(bytes.len()).ok()?.to_be_bytes());
    hash.update(bytes);
    Some(())
}

#[cfg(unix)]
fn reviewable_file_mode(metadata: &Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt as _;
    metadata.permissions().mode() & 0o7777
}

#[cfg(not(unix))]
fn reviewable_file_mode(metadata: &Metadata) -> u32 {
    if metadata.permissions().readonly() {
        0o444
    } else {
        0o666
    }
}

async fn ignored_untracked_paths(root: &Path, changes: &[FileChange]) -> Option<BTreeSet<Vec<u8>>> {
    if changes.is_empty() {
        return Some(BTreeSet::new());
    }

    let mut command = tokio::process::Command::new("git");
    command
        .arg("-C")
        .arg(root)
        .args(["check-ignore", "-z", "--stdin"])
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_INDEX_FILE")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let mut child = command.spawn().ok()?;
    let mut stdin = child.stdin.take()?;
    let mut input = Vec::new();
    for change in changes {
        input.extend_from_slice(change.path.as_bytes());
        input.push(0);
    }

    // Drain stdout while feeding stdin. Writing every path before reading the
    // result can deadlock when a large ignored-path set fills Git's stdout
    // pipe and Git stops consuming stdin.
    let (write_result, output) = tokio::time::timeout(Duration::from_secs(5), async move {
        tokio::join!(
            async move {
                stdin.write_all(&input).await?;
                stdin.shutdown().await
            },
            child.wait_with_output()
        )
    })
    .await
    .ok()?;
    write_result.ok()?;
    let output = output.ok()?;
    // `check-ignore` uses 1 for the successful "no paths matched" result.
    if !output.status.success() && output.status.code() != Some(1) {
        return None;
    }
    let mut ignored = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(<[u8]>::to_vec)
        .collect::<BTreeSet<_>>();
    if !ignored.is_empty() {
        // After `git rm`, a path is absent from the index and `check-ignore`
        // reports it as ignored even though its staged deletion is very much
        // part of the reviewable diff. Preserve every such deletion. If Git
        // cannot classify it, retain all changes conservatively.
        let staged_deletions = staged_deleted_paths(root).await?;
        ignored.retain(|path| !staged_deletions.contains(path));
    }
    Some(ignored)
}

async fn staged_deleted_paths(root: &Path) -> Option<BTreeSet<Vec<u8>>> {
    let mut command = tokio::process::Command::new("git");
    command
        .arg("-C")
        .arg(root)
        .args([
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--diff-filter=D",
            "--no-renames",
            "--relative",
            "--",
        ])
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_INDEX_FILE")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(5), command.output())
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(
        output
            .stdout
            .split(|byte| *byte == 0)
            .filter(|path| !path.is_empty())
            .map(<[u8]>::to_vec)
            .collect(),
    )
}

/// Assess a green turn's file changes against the task contract.
pub(crate) fn assess(
    contract: &TaskContract,
    changes: &[FileChange],
    prompt: &str,
) -> Vec<HygieneFinding> {
    let mut findings = Vec::new();
    if let Some(finding) = unreferenced_creates(contract, changes) {
        findings.push(finding);
    }
    if let Some(finding) = unexpected_dependency_manifest(contract, changes, prompt) {
        findings.push(finding);
    }
    if let Some(finding) = oversized_file(changes) {
        findings.push(finding);
    }
    findings
}

fn unreferenced_creates(contract: &TaskContract, changes: &[FileChange]) -> Option<HygieneFinding> {
    if contract.referenced_paths.is_empty() {
        return None;
    }
    let extras: Vec<&str> = changes
        .iter()
        .filter(|change| change.kind == FileChangeKind::Create)
        .map(|change| change.path.as_str())
        .filter(|path| !path_is_referenced(path, &contract.referenced_paths))
        .collect();
    if extras.len() < UNREFERENCED_CREATE_THRESHOLD {
        return None;
    }
    Some(HygieneFinding {
        reason: format!(
            "narrow mutation named {} but created {} unreferenced files ({})",
            contract.referenced_paths.join(", "),
            extras.len(),
            extras
                .iter()
                .take(6)
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        ),
    })
}

fn path_is_referenced(path: &str, referenced: &[String]) -> bool {
    let normalized = path.replace('\\', "/");
    referenced.iter().any(|named| {
        let named = named.replace('\\', "/");
        normalized == named
            || normalized.starts_with(&format!("{named}/"))
            || named.ends_with('/') && normalized.starts_with(&named)
            || normalized.rsplit('/').next() == Some(named.as_str())
    })
}

fn unexpected_dependency_manifest(
    contract: &TaskContract,
    changes: &[FileChange],
    prompt: &str,
) -> Option<HygieneFinding> {
    // Dependency-manifest changes are suspicious only for a narrow task that
    // named concrete files. Broad feature/integration work routinely needs a
    // client library or runtime package even when the user did not prescribe
    // the implementation detail. Challenging that natural change caused a
    // completed turn to re-enter the repair loop after dozens of inspections.
    if contract.referenced_paths.is_empty() {
        return None;
    }
    if prompt_mentions_dependencies(prompt, contract) {
        return None;
    }
    let manifests: Vec<&str> = changes
        .iter()
        .filter(|change| {
            change.kind == FileChangeKind::Modify && is_dependency_manifest(&change.path)
        })
        .map(|change| change.path.as_str())
        .collect();
    if manifests.is_empty() {
        return None;
    }
    Some(HygieneFinding {
        reason: format!(
            "modified dependency manifest(s) {} without the task asking to add a dependency",
            manifests.join(", ")
        ),
    })
}

fn prompt_mentions_dependencies(prompt: &str, contract: &TaskContract) -> bool {
    let mut blob = prompt.to_ascii_lowercase();
    for line in &contract.acceptance_text {
        blob.push(' ');
        blob.push_str(&line.to_ascii_lowercase());
    }
    [
        "dependenc",
        "add crate",
        "add a crate",
        "add package",
        "add a package",
        "install ",
        "cargo add",
        "npm install",
        "pnpm add",
        "yarn add",
        "pip install",
        "go get",
        "requirements",
    ]
    .iter()
    .any(|needle| blob.contains(needle))
}

fn is_dependency_manifest(path: &str) -> bool {
    let name = path.replace('\\', "/");
    let file = name.rsplit('/').next().unwrap_or(&name);
    matches!(
        file,
        "Cargo.toml"
            | "Cargo.lock"
            | "package.json"
            | "package-lock.json"
            | "pnpm-lock.yaml"
            | "yarn.lock"
            | "pyproject.toml"
            | "go.mod"
            | "go.sum"
            | "requirements.txt"
    )
}

/// Bytes written or removed this turn. `None` when the baseline is unknown
/// (a Modify without `before_len`) — those are not flagged, because an
/// already-large file with a small patch is not a rewrite.
fn rewrite_delta(change: &FileChange) -> Option<u64> {
    match change.kind {
        FileChangeKind::Create => change.after_len,
        FileChangeKind::Modify => {
            let after = change.after_len?;
            Some(after.abs_diff(change.before_len?))
        }
        FileChangeKind::Delete => None,
    }
}

fn oversized_file(changes: &[FileChange]) -> Option<HygieneFinding> {
    changes.iter().find_map(|change| {
        let delta = rewrite_delta(change).filter(|&delta| delta > LARGE_FILE_BYTES)?;
        Some(HygieneFinding {
            reason: format!(
                "{} changed by {} bytes this turn (file is {} bytes) — prefer edit/patch over rewriting large files",
                change.path,
                delta,
                change.after_len.unwrap_or(0)
            ),
        })
    })
}

#[cfg(test)]
#[path = "hygiene_tests.rs"]
mod tests;
