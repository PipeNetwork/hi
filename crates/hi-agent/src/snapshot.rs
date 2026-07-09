//! Lightweight workspace fingerprinting for change detection between snapshots.

/// A lightweight file fingerprint: mtime (nanoseconds) + size in bytes. Two
/// snapshots of the same file compare equal iff the file hasn't been touched.
/// Much cheaper than reading every file's content on every turn. Nanosecond
/// mtime (not seconds) so a same-second, length-preserving edit — e.g. a
/// one-character fix in a rapid multi-turn eval run — isn't missed, which would
/// silently skip the verify gate on exactly the change that needed checking.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FileFingerprint {
    pub(crate) mtime_nanos: u128,
    pub(crate) len: u64,
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
            out.insert(
                rel.to_string_lossy().into_owned(),
                FileFingerprint {
                    mtime_nanos,
                    len: meta.len(),
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
}
