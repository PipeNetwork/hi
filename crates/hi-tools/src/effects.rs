//! Exact workspace effect accounting for opaque process tools.
//!
//! Shell commands cannot declare their targets up front, so their effects are
//! measured by comparing complete workspace fingerprints before and after the
//! process. Ignore files are deliberately disabled: ignored configuration is
//! still user data. Only known VCS, runtime-state, dependency, and generated
//! trees are pruned.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};

use crate::{FileChange, FileChangeKind, ToolEffects};

pub(crate) type WorkspaceSnapshot = BTreeMap<String, WorkspaceEntry>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryKind {
    File,
    Symlink,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceEntry {
    kind: EntryKind,
    digest: String,
    len: u64,
    mode: u32,
}

/// Fingerprint every relevant workspace file without consulting ignore rules.
///
/// The walk and hashing run off the async executor. Any traversal, metadata,
/// read, or special-node error is returned to the caller; a process outcome
/// must never claim "no changes" merely because effect inspection failed.
pub(crate) async fn workspace_snapshot(
    root: &Path,
    state_root: &Path,
) -> Result<WorkspaceSnapshot> {
    let root = root.to_path_buf();
    let state_root = state_root.to_path_buf();
    tokio::task::spawn_blocking(move || workspace_snapshot_blocking(&root, &state_root))
        .await
        .context("workspace effect snapshot task failed")?
}

fn workspace_snapshot_blocking(root: &Path, state_root: &Path) -> Result<WorkspaceSnapshot> {
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalizing workspace root {}", root.display()))?;
    let state_root = if state_root.is_absolute() {
        state_root.to_path_buf()
    } else {
        root.join(state_root)
    };
    let state_root = state_root.canonicalize().unwrap_or(state_root);
    ensure!(
        state_root != root && !root.starts_with(&state_root),
        "runtime state root cannot equal or contain workspace root {}",
        root.display()
    );
    let state_relative = state_root.strip_prefix(&root).ok().map(Path::to_path_buf);
    let filter_root = root.clone();
    let filter_state = state_relative.clone();
    let mut snapshot = BTreeMap::new();

    for entry in ignore::WalkBuilder::new(&root)
        .hidden(false)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .ignore(false)
        .parents(false)
        .follow_links(false)
        .filter_entry(move |entry| {
            let relative = entry.path().strip_prefix(&filter_root).ok();
            if relative.is_some_and(|path| path.starts_with(".hi/state"))
                || filter_state
                    .as_ref()
                    .is_some_and(|state| relative.is_some_and(|path| path.starts_with(state)))
            {
                return false;
            }
            relative.is_none_or(|path| {
                path.as_os_str().is_empty()
                    || (!is_vcs_name(entry.file_name())
                        && !(entry
                            .file_type()
                            .is_some_and(|file_type| file_type.is_dir())
                            && is_generated_directory_name(entry.file_name())))
            })
        })
        .build()
    {
        let entry = entry.with_context(|| format!("walking workspace {}", root.display()))?;
        let path = entry.path();
        let metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("reading workspace entry {}", path.display()))?;
        let file_type = metadata.file_type();
        if file_type.is_dir() {
            continue;
        }

        let relative = path
            .strip_prefix(&root)
            .with_context(|| format!("workspace walker escaped root at {}", path.display()))?;
        let key = portable_path(relative);
        let mode = file_mode(&metadata);
        let workspace_entry = if file_type.is_file() {
            let mut file = std::fs::File::open(path)
                .with_context(|| format!("opening {} for effect accounting", path.display()))?;
            let mut hasher = Sha256::new();
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                let read = file
                    .read(&mut buffer)
                    .with_context(|| format!("reading {} for effect accounting", path.display()))?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            WorkspaceEntry {
                kind: EntryKind::File,
                digest: format!("sha256:{:x}", hasher.finalize()),
                len: metadata.len(),
                mode,
            }
        } else if file_type.is_symlink() {
            let target = std::fs::read_link(path)
                .with_context(|| format!("reading symlink {}", path.display()))?;
            let bytes = target.as_os_str().as_encoded_bytes();
            WorkspaceEntry {
                kind: EntryKind::Symlink,
                digest: format!("sha256:{:x}", Sha256::digest(bytes)),
                len: bytes.len() as u64,
                mode,
            }
        } else {
            bail!(
                "cannot inspect special workspace entry {} for process effects",
                path.display()
            );
        };
        snapshot.insert(key, workspace_entry);
    }
    Ok(snapshot)
}

pub(crate) fn changes_between(
    before: &WorkspaceSnapshot,
    after: &WorkspaceSnapshot,
) -> Vec<FileChange> {
    let paths: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    paths
        .into_iter()
        .filter_map(|path| {
            let before_entry = before.get(path);
            let after_entry = after.get(path);
            if before_entry == after_entry {
                return None;
            }
            let kind = match (before_entry, after_entry) {
                (None, Some(_)) => FileChangeKind::Create,
                (Some(_), None) => FileChangeKind::Delete,
                (Some(_), Some(_)) => FileChangeKind::Modify,
                (None, None) => return None,
            };
            Some(FileChange {
                path: path.clone(),
                kind,
                before_digest: before_entry.map(|entry| entry.digest.clone()),
                after_digest: after_entry.map(|entry| entry.digest.clone()),
                before_len: before_entry.map(|entry| entry.len),
                after_len: after_entry.map(|entry| entry.len),
                before_mode: before_entry.map(|entry| entry.mode),
                after_mode: after_entry.map(|entry| entry.mode),
            })
        })
        .collect()
}

pub(crate) fn process_effects(
    before: &WorkspaceSnapshot,
    after: &WorkspaceSnapshot,
) -> ToolEffects {
    let file_changes = changes_between(before, after);
    ToolEffects {
        mutation_attempted: true,
        mutation_applied: !file_changes.is_empty(),
        file_changes,
    }
}

fn is_vcs_name(name: &std::ffi::OsStr) -> bool {
    matches!(name.to_str(), Some(".git" | ".hg" | ".svn" | ".jj"))
}

fn is_generated_directory_name(name: &std::ffi::OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(
            "target"
                | "node_modules"
                | ".venv"
                | "venv"
                | "vendor"
                | "models"
                | ".cache"
                | "dist"
                | "build"
                | ".next"
                | ".turbo"
                | "coverage"
                | "__pycache__"
                | ".pytest_cache"
                | ".mypy_cache"
                | ".ruff_cache"
                | "hi-test-scratch"
        )
    )
}

fn portable_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(unix)]
fn file_mode(metadata: &std::fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o7777
}

#[cfg(not(unix))]
fn file_mode(metadata: &std::fs::Metadata) -> u32 {
    if metadata.permissions().readonly() {
        0o444
    } else {
        0o666
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ignored_files_are_measured_while_generated_and_state_trees_are_pruned() {
        let root = std::env::temp_dir().join(format!("hi-tool-effects-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let state = root.join("runtime-state");
        std::fs::create_dir_all(root.join("target/debug")).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        std::fs::write(root.join(".gitignore"), "ignored.env\n").unwrap();
        std::fs::write(root.join("ignored.env"), "before\n").unwrap();
        std::fs::write(root.join("build"), "source script\n").unwrap();
        std::fs::write(root.join("target/debug/artifact"), "generated\n").unwrap();
        std::fs::write(state.join("manifest"), "runtime\n").unwrap();

        let before = workspace_snapshot(&root, &state).await.unwrap();
        std::fs::write(root.join("ignored.env"), "after\n").unwrap();
        std::fs::write(root.join("target/debug/artifact"), "changed\n").unwrap();
        std::fs::write(state.join("manifest"), "changed\n").unwrap();
        let after = workspace_snapshot(&root, &state).await.unwrap();
        let changes = changes_between(&before, &after);

        assert_eq!(changes.len(), 1, "changes: {changes:?}");
        assert_eq!(changes[0].path, "ignored.env");
        assert_eq!(changes[0].kind, FileChangeKind::Modify);
        assert!(
            after.contains_key("build"),
            "only build directories are pruned"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn special_nodes_fail_effect_inspection() {
        use std::os::unix::net::UnixListener;

        let root =
            std::env::temp_dir().join(format!("hi-tool-effects-special-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let socket = UnixListener::bind(root.join("live.sock")).unwrap();
        let error = workspace_snapshot(&root, &root.join("state"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("special workspace entry"));
        drop(socket);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn state_root_cannot_hide_the_entire_workspace() {
        let root = std::env::temp_dir().join(format!(
            "hi-tool-effects-invalid-state-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("source.txt"), "visible\n").unwrap();

        let error = workspace_snapshot(&root, &root).await.unwrap_err();

        assert!(error.to_string().contains("cannot equal or contain"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn changes_include_create_delete_and_mode_only_modify() {
        use std::os::unix::fs::PermissionsExt;

        let root =
            std::env::temp_dir().join(format!("hi-tool-effects-kinds-{}", std::process::id()));
        let state = std::env::temp_dir().join(format!(
            "hi-tool-effects-kinds-state-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("delete.txt"), "gone\n").unwrap();
        std::fs::write(root.join("mode.sh"), "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(root.join("mode.sh"), std::fs::Permissions::from_mode(0o644))
            .unwrap();
        let before = workspace_snapshot(&root, &state).await.unwrap();

        std::fs::remove_file(root.join("delete.txt")).unwrap();
        std::fs::write(root.join("create.txt"), "new\n").unwrap();
        std::fs::set_permissions(root.join("mode.sh"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        let after = workspace_snapshot(&root, &state).await.unwrap();
        let changes = changes_between(&before, &after);

        assert_eq!(changes.len(), 3, "changes: {changes:?}");
        assert_eq!(changes[0].path, "create.txt");
        assert_eq!(changes[0].kind, FileChangeKind::Create);
        assert_eq!(changes[1].path, "delete.txt");
        assert_eq!(changes[1].kind, FileChangeKind::Delete);
        assert_eq!(changes[2].path, "mode.sh");
        assert_eq!(changes[2].kind, FileChangeKind::Modify);
        assert_eq!(changes[2].before_digest, changes[2].after_digest);
        assert_eq!(changes[2].before_mode, Some(0o644));
        assert_eq!(changes[2].after_mode, Some(0o755));
        let _ = std::fs::remove_dir_all(root);
    }
}
