//! Lightweight workspace fingerprinting for change detection between snapshots.

/// A lightweight file fingerprint: mtime (nanoseconds) + size in bytes. Two
/// snapshots of the same file compare equal iff the file hasn't been touched.
/// Much cheaper than reading every file's content on every turn. Nanosecond
/// mtime (not seconds) so a same-second, length-preserving edit — e.g. a
/// one-character fix in a rapid multi-turn eval run — isn't missed, which would
/// silently skip the verify gate on exactly the change that needed checking.
///
/// That nanosecond guard only holds on filesystems that *report* sub-second
/// mtime; coarse-granularity ones (network mounts, FAT, some CI overlays)
/// truncate to whole seconds, reopening the same blind spot. `content_hash`
/// closes it: computed only when the mtime lands on a whole second (the coarse
/// signal), so the extra content read almost never runs on APFS/ext4.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FileFingerprint {
    pub(crate) mtime_nanos: u128,
    pub(crate) len: u64,
    pub(crate) content_hash: Option<u64>,
}

pub(crate) async fn workspace_snapshot(
    dir: &std::path::Path,
) -> std::collections::BTreeMap<String, FileFingerprint> {
    // Use the `ignore` crate (same as the list/grep tools) to respect
    // .gitignore, global gitignore, and parent .gitignore files. This avoids
    // walking node_modules, .venv, target, vendor, Pods, etc. — a massive win
    // for repos with large dependency trees. The walk is synchronous but fast
    // (no per-entry async overhead); we run it on a blocking-pool thread.
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut out = std::collections::BTreeMap::new();
        for entry in ignore::WalkBuilder::new(&dir)
            .hidden(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .ignore(true)
            .parents(true)
            // Prune VCS metadata and common generated dependency/build trees.
            // Fresh temp projects often have no .gitignore yet, so relying only
            // on ignore rules makes `cargo test` or package installs show
            // hundreds of generated files as changed user work.
            .filter_entry(|e| {
                !matches!(
                    e.file_name().to_str(),
                    Some(
                        ".git"
                            | ".hg"
                            | ".svn"
                            | ".jj"
                            // hi's own state dir (goal/memory/session
                            // persistence). The agent writing its own
                            // bookkeeping mid-turn is not user work and must
                            // not trip the verify gate or changed-files lists.
                            | ".hi"
                            | "target"
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
                    )
                )
            })
            .build()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Ok(rel) = path.strip_prefix(&dir) else {
                continue;
            };
            let Ok(meta) = std::fs::metadata(path) else {
                continue;
            };
            let mtime_nanos = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            // On a coarse-granularity filesystem `modified()` truncates to whole
            // seconds, so a same-second, length-preserving edit leaves mtime+len
            // unchanged and would silently skip the verify gate. When the mtime is
            // a whole second (the coarse signal), fall back to a content hash so
            // the edit is still seen. On fine clocks the sub-second part is ~never
            // zero, so this read almost never runs.
            let content_hash = if mtime_nanos % 1_000_000_000 == 0 {
                std::fs::read(path).ok().map(|bytes| {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    bytes.hash(&mut h);
                    h.finish()
                })
            } else {
                None
            };
            out.insert(
                rel.to_string_lossy().into_owned(),
                FileFingerprint {
                    mtime_nanos,
                    len: meta.len(),
                    content_hash,
                },
            );
        }
        out
    })
    .await
    .unwrap_or_default()
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
    pub(crate) async fn get(&mut self) -> std::collections::BTreeMap<String, FileFingerprint> {
        if let Some(cache) = &self.cached {
            return cache.clone();
        }
        let snap = workspace_snapshot(std::path::Path::new(".")).await;
        self.cached = Some(snap.clone());
        snap
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

        let snapshot = workspace_snapshot(&dir).await;
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

        let snapshot = workspace_snapshot(&dir).await;
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
        let before = workspace_snapshot(&dir).await;
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
        let after = workspace_snapshot(&dir).await;

        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            changed_files_between(&before, &after),
            vec!["a.txt".to_string()],
            "same-second, same-length edit detected via content hash"
        );
    }
}
