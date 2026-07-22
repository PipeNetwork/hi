//! Lightweight workspace fingerprinting for change detection between snapshots.

use std::hash::Hasher;
use std::io::Read;

use anyhow::{Context, Result};

/// A lightweight file fingerprint: mtime (nanoseconds) + size in bytes. Two
/// snapshots of the same file compare equal iff the file hasn't been touched.
/// Much cheaper than reading every file's content on every turn. Nanosecond
/// mtime (not seconds) so a same-second, length-preserving edit — e.g. a
/// one-character fix in a rapid multi-turn eval run — isn't missed, which would
/// silently skip the verify gate on exactly the change that needed checking.
///
/// `content_hash` is always computed so coarse timestamps, length-preserving
/// edits, and timestamp restoration cannot hide a changed verification input.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FileFingerprint {
    pub(crate) mtime_nanos: u128,
    pub(crate) len: u64,
    pub(crate) content_hash: Option<u64>,
}

pub(crate) async fn workspace_snapshot(
    dir: &std::path::Path,
) -> Result<std::collections::BTreeMap<String, FileFingerprint>> {
    workspace_snapshot_with(dir, true).await
}

/// Fast mtime+size snapshot for per-stage mutation detection. Skips content
/// hashing — verify stages must not rewrite sources, and a touch that preserves
/// length still updates mtime on normal filesystems. Turn baselines still use
/// [`workspace_snapshot`] (full content hash).
pub(crate) async fn workspace_snapshot_meta(
    dir: &std::path::Path,
) -> Result<std::collections::BTreeMap<String, FileFingerprint>> {
    workspace_snapshot_with(dir, false).await
}

async fn workspace_snapshot_with(
    dir: &std::path::Path,
    hash_contents: bool,
) -> Result<std::collections::BTreeMap<String, FileFingerprint>> {
    // This is a verification boundary, so `.gitignore` must not hide inputs
    // such as `.env` or generated configuration. Large dependency/build trees
    // are pruned explicitly below. The walk is synchronous and runs on the
    // blocking pool.
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<_> {
        let mut out = std::collections::BTreeMap::new();
        let filter_root = dir.clone();
        for entry in ignore::WalkBuilder::new(&dir)
            .hidden(false)
            .git_ignore(false)
            .git_global(false)
            .git_exclude(false)
            .ignore(false)
            .parents(false)
            // Prune VCS metadata and common generated dependency/build trees.
            // Fresh temp projects often have no .gitignore yet, so relying only
            // on ignore rules makes `cargo test` or package installs show
            // hundreds of generated files as changed user work.
            .filter_entry(move |e| {
                let relative = e.path().strip_prefix(&filter_root).ok();
                let runtime_state =
                    relative.is_some_and(|relative| relative.starts_with(".hi/state"));
                let weight_cache = relative.is_some_and(|relative| {
                    let mut components =
                        relative
                            .components()
                            .filter_map(|component| match component {
                                std::path::Component::Normal(name) => name.to_str(),
                                _ => None,
                            });
                    matches!(
                        (components.next(), components.next()),
                        (Some("models"), _) | (Some(".hi"), Some("models"))
                    )
                });
                !runtime_state
                    && !weight_cache
                    && !matches!(
                        e.file_name().to_str(),
                        Some(
                            ".git"
                                | ".hg"
                                | ".svn"
                                | ".jj"
                                | "target"
                                | "node_modules"
                                | ".venv"
                                | "venv"
                                | "vendor"
                                | ".cache"
                                | "dist"
                                | "build"
                                | ".next"
                                | ".turbo"
                                | "coverage"
                                | "hi-test-scratch"
                        )
                    )
            })
            .build()
        {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error)
                    if error
                        .io_error()
                        .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
                {
                    continue;
                }
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("walking workspace {}", dir.display()));
                }
            };
            let path = entry.path();
            let meta = match std::fs::symlink_metadata(path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("reading workspace entry {}", path.display()));
                }
            };
            if meta.is_dir() {
                continue;
            }
            if !meta.is_file() && !meta.file_type().is_symlink() {
                anyhow::bail!(
                    "cannot fingerprint special workspace entry {}",
                    path.display()
                );
            }
            let rel = path
                .strip_prefix(&dir)
                .with_context(|| format!("workspace walker escaped root at {}", path.display()))?;
            let mtime_nanos = meta
                .modified()
                .with_context(|| format!("reading modification time for {}", path.display()))?
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let len = if meta.file_type().is_symlink() {
                std::fs::read_link(path)
                    .with_context(|| format!("reading symlink {}", path.display()))?
                    .as_os_str()
                    .as_encoded_bytes()
                    .len() as u64
            } else {
                meta.len()
            };
            let content_hash = if !hash_contents {
                None
            } else if meta.file_type().is_symlink() {
                let target = std::fs::read_link(path)
                    .with_context(|| format!("reading symlink {}", path.display()))?;
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                hash.write(target.as_os_str().as_encoded_bytes());
                Some(hash.finish())
            } else {
                let mut file = std::fs::File::open(path)
                    .with_context(|| format!("opening {} for fingerprinting", path.display()))?;
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                let mut buffer = [0_u8; 64 * 1024];
                loop {
                    let read = file.read(&mut buffer).with_context(|| {
                        format!("reading {} for fingerprinting", path.display())
                    })?;
                    if read == 0 {
                        break;
                    }
                    hash.write(&buffer[..read]);
                }
                Some(hash.finish())
            };
            out.insert(
                rel.to_string_lossy().into_owned(),
                FileFingerprint {
                    mtime_nanos,
                    len,
                    content_hash,
                },
            );
        }
        Ok(out)
    })
    .await
    .context("workspace snapshot task failed")?
}

pub(crate) fn changed_files_between(
    before: &std::collections::BTreeMap<String, FileFingerprint>,
    after: &std::collections::BTreeMap<String, FileFingerprint>,
) -> Vec<String> {
    let mut files = std::collections::BTreeSet::new();
    for path in before.keys() {
        if before.get(path) != after.get(path) {
            files.insert(path.clone());
        }
    }
    for path in after.keys() {
        if before.get(path) != after.get(path) {
            files.insert(path.clone());
        }
    }
    files.into_iter().collect()
}

/// A cached workspace snapshot, invalidated by any mutating tool call (or
/// `undo`). Wraps the raw `Option<BTreeMap>` so the agent doesn't hold a
/// bare cache field that's easy to forget to invalidate — the type makes
/// "take a fresh snapshot after the tree changed" the obvious operation.
#[derive(Default)]
pub(crate) struct SnapshotCache {
    cached: Option<std::collections::BTreeMap<String, FileFingerprint>>,
}

impl SnapshotCache {
    /// Get the workspace snapshot, using the cached version when still valid.
    /// The cache is valid until [`invalidate`] is called.
    ///
    /// [`invalidate`]: Self::invalidate
    pub(crate) async fn get(
        &mut self,
        root: &std::path::Path,
    ) -> Result<std::collections::BTreeMap<String, FileFingerprint>> {
        if let Some(cache) = &self.cached {
            return Ok(cache.clone());
        }
        let snap = workspace_snapshot(root).await?;
        self.cached = Some(snap.clone());
        Ok(snap)
    }

    /// Invalidate the cache — call after any operation that may change the
    /// working tree (a mutating tool, or `/undo` restoring a checkpoint), so
    /// the next [`get`] re-walks instead of returning stale fingerprints.
    ///
    /// [`get`]: Self::get
    pub(crate) fn invalidate(&mut self) {
        self.cached = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn workspace_snapshot_ignores_generated_build_directories_without_gitignore() {
        let dir =
            std::env::temp_dir().join(format!("hi-snapshot-generated-dirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(dir.join("target/debug")).unwrap();
        std::fs::write(dir.join("src/lib.rs"), "pub fn ok() {}\n").unwrap();
        std::fs::write(dir.join("target/debug/generated.o"), "artifact\n").unwrap();

        let snapshot = workspace_snapshot(&dir).await.unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert!(snapshot.contains_key("src/lib.rs"));
        assert!(
            snapshot.keys().all(|path| !path.starts_with("target/")),
            "snapshot should ignore target artifacts: {:?}",
            snapshot.keys().collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn workspace_snapshot_ignores_heavy_untracked_directories_without_gitignore() {
        let dir = std::env::temp_dir().join(format!(
            "hi-snapshot-heavy-untracked-dirs-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        for path in [
            "models/model.bin",
            ".cache/tool/output",
            "dist/app.js",
            "build/output",
            ".next/cache",
            ".turbo/cache",
            "coverage/lcov.info",
        ] {
            let path = dir.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "generated\n").unwrap();
        }
        std::fs::write(dir.join("src/lib.rs"), "pub fn ok() {}\n").unwrap();

        let snapshot = workspace_snapshot(&dir).await.unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert!(snapshot.contains_key("src/lib.rs"));
        assert!(
            snapshot.keys().all(|path| ![
                "models/",
                ".cache/",
                "dist/",
                "build/",
                ".next/",
                ".turbo/",
                "coverage/",
            ]
            .iter()
            .any(|prefix| path.starts_with(prefix))),
            "snapshot should ignore heavy generated dirs: {:?}",
            snapshot.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn content_hash_distinguishes_same_mtime_same_len() {
        // A same-second, length-preserving edit on a coarse-mtime filesystem
        // leaves mtime+len identical; the content hash is what tells them apart.
        let fp = |hash: Option<u64>| FileFingerprint {
            mtime_nanos: 1_700_000_000_000_000_000,
            len: 6,
            content_hash: hash,
        };
        let mut before = std::collections::BTreeMap::new();
        before.insert("a.txt".to_string(), fp(Some(111)));
        let mut after = std::collections::BTreeMap::new();
        after.insert("a.txt".to_string(), fp(Some(222)));
        assert_eq!(
            changed_files_between(&before, &after),
            vec!["a.txt".to_string()]
        );
        // Same fingerprint (same hash) → no change reported.
        assert!(changed_files_between(&before, &before.clone()).is_empty());
    }

    #[tokio::test]
    async fn same_second_same_length_edit_is_caught_by_content_hash() {
        use std::time::{Duration, UNIX_EPOCH};
        let dir = std::env::temp_dir().join(format!("hi-snapshot-coarse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.txt");

        // Simulate a coarse-granularity filesystem: force a whole-second mtime.
        let whole_second = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        std::fs::write(&file, "aaaaa\n").unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&file)
            .unwrap()
            .set_modified(whole_second)
            .unwrap();
        let before = workspace_snapshot(&dir).await.unwrap();
        assert!(
            before["a.txt"].content_hash.is_some(),
            "a whole-second mtime is hashed"
        );

        // Edit with the SAME length and reset the SAME whole-second mtime —
        // invisible to mtime+len alone, but the content hash must still flag it.
        std::fs::write(&file, "bbbbb\n").unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&file)
            .unwrap()
            .set_modified(whole_second)
            .unwrap();
        let after = workspace_snapshot(&dir).await.unwrap();

        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            changed_files_between(&before, &after),
            vec!["a.txt".to_string()],
            "same-second, same-length edit detected via content hash"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn special_workspace_entries_surface_snapshot_errors() {
        let dir = std::env::temp_dir().join(format!("hi-snapshot-special-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let _socket = std::os::unix::net::UnixListener::bind(dir.join("service.sock")).unwrap();

        let error = workspace_snapshot(&dir).await.unwrap_err();
        assert!(error.to_string().contains("special workspace entry"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
