use std::collections::HashMap;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

use anyhow::{Context, Result, bail};

/// Validate that a path is inside the workspace root (cwd by default). Returns
/// the canonicalized absolute path if safe, or an error explaining why not.
/// Set `HI_NO_PATH_GUARD=1` to disable (not recommended — the model can then
/// read/write any file on the system).
pub(crate) fn validate_workspace_path(path: &str) -> Result<std::path::PathBuf> {
    if std::env::var_os("HI_NO_PATH_GUARD").is_some() {
        return Ok(Path::new(path).to_path_buf());
    }
    let cwd = std::env::current_dir().context("determining working directory")?;
    let target = Path::new(path);
    // If absolute, canonicalize and check containment. If relative, join to cwd.
    let resolved = if target.is_absolute() {
        target.to_path_buf()
    } else {
        cwd.join(target)
    };
    // For paths that exist, canonicalize to resolve symlinks and `..`. For
    // paths that don't exist yet (a new file being written), canonicalize()
    // fails — so we canonicalize the *parent* directory (which usually exists)
    // and re-join the filename. This resolves symlinks on the parent so a
    // symlink inside the workspace pointing outside (e.g. `./external -> /etc`)
    // can't be used to write `external/new_file` to `/etc/new_file` — the
    // canonicalized parent would be `/etc`, which fails the containment check.
    // If the parent also doesn't exist (nested new directories), fall back to
    // *lexical* normalization (resolve `.`/`..` without touching the filesystem)
    // so `..` segments can't escape the workspace via a not-yet-existing path.
    let canonical = resolved
        .canonicalize()
        .unwrap_or_else(|_| canonicalize_via_parent(&resolved));
    let canonical_cwd = cwd.canonicalize().unwrap_or(cwd.clone());
    if canonical.starts_with(&canonical_cwd) {
        return Ok(canonical);
    }
    // Allow /tmp and macOS /var/folders paths (scratch files, pipes). On macOS,
    // /tmp symlinks to /private/tmp and /var/folders to /private/var/folders,
    // so canonicalize() resolves them.
    if canonical.starts_with("/tmp/")
        || canonical.starts_with("/private/tmp/")
        || canonical.starts_with("/var/folders/")
        || canonical.starts_with("/private/var/folders/")
    {
        return Ok(canonical);
    }
    bail!(
        "path '{}' is outside the workspace ({}). \
         Set HI_NO_PATH_GUARD=1 to allow out-of-workspace paths.",
        path,
        canonical_cwd.display()
    );
}

/// Canonicalize a not-yet-existing path by resolving its nearest existing
/// ancestor and re-joining the not-yet-existing tail. Canonicalizing the
/// deepest existing ancestor resolves symlinks in *every* existing component,
/// so a symlink inside the workspace pointing outside can't be used to escape —
/// even through a not-yet-existing intermediate directory. For example, with
/// `ws/link -> /etc` and target `ws/link/new/passwd`: canonicalizing only the
/// immediate parent (`ws/link/new`) fails because `new` doesn't exist, but
/// walking up to `ws/link` resolves the symlink, so containment sees
/// `/etc/new/passwd` and refuses it. Falls back to lexical normalization only
/// if no ancestor canonicalizes (should not happen — the root always does).
fn canonicalize_via_parent(path: &Path) -> std::path::PathBuf {
    // Normalize `.`/`..` first so the ancestor walk is a clean climb to root.
    let abs = lexical_abs(path);
    let mut existing = abs.as_path();
    // File names of the not-yet-existing tail, collected deepest-first.
    let mut tail: Vec<&std::ffi::OsStr> = Vec::new();
    loop {
        if let Ok(canonical) = existing.canonicalize() {
            let mut resolved = canonical;
            // Re-append the tail (which was collected deepest-first).
            for name in tail.iter().rev() {
                resolved.push(name);
            }
            return resolved;
        }
        match existing.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                if let Some(name) = existing.file_name() {
                    tail.push(name);
                }
                existing = parent;
            }
            _ => break,
        }
    }
    abs
}

/// Lexically normalize a path to an absolute form with no `.` or `..` segments
/// — without touching the filesystem (so it works for not-yet-existing paths).
/// Symlinks are NOT resolved (that requires filesystem access); this only
/// collapses `.`/`..` components lexically. Used as the fallback when
/// `canonicalize()` fails on a new file, and to produce a stable cache key.
pub(crate) fn lexical_abs(path: &Path) -> std::path::PathBuf {
    use std::path::Component;

    // Make it absolute first (relative paths are joined to cwd).
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| Path::new("/").to_path_buf())
            .join(path)
    };

    let mut out = std::path::PathBuf::new();
    for comp in abs.components() {
        match comp {
            Component::CurDir => {} // skip "."
            Component::ParentDir => {
                // Pop the last normal component (if any). If we're at root,
                // `..` is a no-op (can't go above root).
                out.pop();
            }
            Component::RootDir => out.push("/"),
            Component::Normal(s) => out.push(s),
            Component::Prefix(p) => out.push(p.as_os_str()),
        }
    }
    out
}

/// Produce a stable cache key for `path`: the lexically-normalized absolute
/// form. This ensures `read("src/foo.rs")`, `read("./src/foo.rs")`, and
/// `read("src/../src/foo.rs")` all share one cache entry, and that
/// invalidation after an edit hits the same key a subsequent read looks up.
/// Falls back to the raw path if the cwd is unavailable.
pub(crate) fn cache_key(path: &str) -> String {
    lexical_abs(Path::new(path)).to_string_lossy().into_owned()
}

/// VCS metadata directories that must never reach the model. We walk with
/// `hidden(false)` so the agent can see useful dotfiles (`.github/`,
/// `.env.example`, `.cargo/config.toml`, ...), but these internal directories are
/// large, mostly binary, and leak repository internals (loose/packed objects,
/// refs, reflogs, config). Used as a `WalkBuilder::filter_entry` predicate,
/// which prunes the whole subtree so we never even descend into them.
pub(crate) fn is_vcs_metadata_dir(entry: &ignore::DirEntry) -> bool {
    matches!(
        entry.file_name().to_str(),
        Some(".git" | ".hg" | ".svn" | ".jj")
    )
}

/// Maximum number of cached file reads. Beyond this, the least
/// recently used entry is evicted (LRU) — the cache refills lazily
/// on the next re-read — rather than clearing entirely, so the hot
/// working set survives a large-repo scan.
pub(crate) const READ_CACHE_MAX: usize = 50;

/// Per-turn cache of file reads, so re-reading the same file (common when the
/// model is orienting) hits memory instead of disk. Cleared between turns, and
/// bounded to [`READ_CACHE_MAX`] entries to avoid unbounded memory growth,
/// via LRU eviction so overflow keeps the hot working set (the old
/// behavior cleared the whole cache on overflow — a performance cliff
/// when the model read >50 files).
pub(crate) static READ_CACHE: LazyLock<Mutex<ReadCache>> =
    LazyLock::new(|| Mutex::new(ReadCache::new()));

/// LRU-ordered file-read cache: a HashMap for O(1) lookup
/// paired with a VecDeque tracking access order. On `get`, the key is
/// promoted to the back of the deque (most-recently-used); on `insert`, it's
/// pushed to the back; on overflow, the front (least-recently-used) is
/// evicted; on `remove`, the key is dropped from the deque too. All
/// operations are O(1) amortized. The deque is bounded to
/// [`READ_CACHE_MAX`] entries, matching the map.
pub struct ReadCache {
    map: HashMap<String, String>,
    order: std::collections::VecDeque<String>,
}

impl ReadCache {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    /// Get a cached entry, promoting it to most-recently-used.
    pub fn get(&mut self, key: &str) -> Option<&String> {
        if let Some(val) = self.map.get(key) {
            // Promote to back (MRU) — remove from current position, push to end
            self.order.retain(|k| k != key);
            self.order.push_back(key.to_string());
            Some(val)
        } else {
            None
        }
    }

    /// Insert an entry, evicting the LRU (front) on overflow.
    pub fn insert(&mut self, key: String, value: String) {
        if self.map.contains_key(&key) {
            // Already present — update value and promote
            self.map.insert(key.clone(), value);
            self.order.retain(|k| k != &key);
            self.order.push_back(key);
        } else {
            if self.map.len() >= READ_CACHE_MAX {
                // Evict least-recently-used (front of deque)
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                }
            }
            self.map.insert(key.clone(), value);
            self.order.push_back(key);
        }
    }

    /// Remove an entry (invalidate after a write/edit).
    pub fn remove(&mut self, key: &str) {
        if self.map.remove(key).is_some() {
            self.order.retain(|k| k != key);
        }
    }

    /// Clear all entries (between turns).
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

/// Clear the per-turn read cache. Call at the start of each turn.
pub fn clear_read_cache() {
    if let Ok(mut cache) = READ_CACHE.lock() {
        cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{ReadCache, cache_key, lexical_abs};
    use std::path::Path;

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut cache = ReadCache::new();
        // Fill to capacity (READ_CACHE_MAX = 50).
        for i in 0..50 {
            cache.insert(format!("file{i}"), format!("content{i}"));
        }
        assert_eq!(cache.map.len(), 50);

        // Access file0 — promotes it to MRU.
        let _ = cache.get("file0");

        // Insert one more — should evict file1 (LRU), not file0.
        cache.insert("file50".into(), "content50".into());
        assert_eq!(cache.map.len(), 50);
        assert!(
            !cache.map.contains_key("file1"),
            "file1 should be evicted (LRU)"
        );
        assert!(
            cache.map.contains_key("file0"),
            "file0 should survive — it was accessed recently"
        );
        assert!(cache.map.contains_key("file50"));
    }

    #[test]
    fn lru_get_promotes_to_mru() {
        let mut cache = ReadCache::new();
        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());

        // Access "a" — promote it.
        assert_eq!(cache.get("a").map(|s| s.as_str()), Some("1"));

        // Insert enough to evict "b" (LRU), keeping "a" (MRU).
        for i in 0..49 {
            cache.insert(format!("x{i}"), format!("v{i}"));
        }
        assert!(
            cache.map.contains_key("a"),
            "a should survive — it was accessed after b"
        );
    }

    #[test]
    fn lru_remove_drops_from_order() {
        let mut cache = ReadCache::new();
        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());
        cache.remove("a");
        assert!(!cache.map.contains_key("a"));
        // After removing "a", inserting up to capacity should not try to evict it.
        for i in 0..49 {
            cache.insert(format!("x{i}"), format!("v{i}"));
        }
        assert_eq!(cache.map.len(), 50);
        assert!(cache.map.contains_key("b"));
    }

    #[test]
    fn lru_insert_updates_existing() {
        let mut cache = ReadCache::new();
        cache.insert("a".into(), "old".into());
        cache.insert("a".into(), "new".into());
        assert_eq!(cache.map.len(), 1);
        assert_eq!(cache.get("a").map(|s| s.as_str()), Some("new"));
    }

    #[test]
    fn lru_clear_empties() {
        let mut cache = ReadCache::new();
        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());
        cache.clear();
        assert_eq!(cache.map.len(), 0);
        assert!(cache.order.is_empty());
    }

    #[test]
    fn lexical_abs_collapses_dotdot() {
        // `..` after a normal component cancels it.
        let p = lexical_abs(Path::new("/a/b/../c"));
        assert_eq!(p, Path::new("/a/c"));
        // `.` is dropped.
        let p = lexical_abs(Path::new("/a/./b"));
        assert_eq!(p, Path::new("/a/b"));
        // `..` at root is a no-op.
        let p = lexical_abs(Path::new("/.."));
        assert_eq!(p, Path::new("/"));
        // Multiple `..` cancel multiple components.
        let p = lexical_abs(Path::new("/a/b/c/../../d"));
        assert_eq!(p, Path::new("/a/d"));
    }

    #[test]
    fn cache_key_normalizes_equivalent_paths() {
        // These three all refer to the same file and must share a cache key.
        let k1 = cache_key("src/foo.rs");
        let k2 = cache_key("./src/foo.rs");
        let k3 = cache_key("src/../src/foo.rs");
        assert_eq!(k1, k2, "leading ./ should not change the key");
        assert_eq!(k1, k3, "redundant ../ should not change the key");
        // The key is absolute (joined to cwd).
        assert!(k1.starts_with('/'), "cache key should be absolute");
    }

    #[cfg(unix)]
    #[test]
    fn canonicalize_via_parent_resolves_symlink() {
        // The fix for symlink escape: canonicalize_via_parent resolves the
        // parent directory's symlinks so a symlink inside the workspace pointing
        // outside can't be used to write a not-yet-existing file through it.
        // The symlink target must NOT be under /tmp or /var/folders (those are
        // allowlisted as scratch paths), so we use a temp dir under the user's
        // home instead.
        use super::canonicalize_via_parent;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."));
        let workspace = base.join(format!(".hi-symlink-ws-{stamp}"));
        let outside = base.join(format!(".hi-symlink-out-{stamp}"));
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let link = workspace.join("escape");
        std::os::unix::fs::symlink(&outside, &link).unwrap();

        // A not-yet-existing file through the symlink: canonicalize_via_parent
        // resolves the parent (the symlink → outside) and joins the filename.
        let target = link.join("new_file.txt");
        let resolved = canonicalize_via_parent(&target);

        // The resolved path must be under `outside`, not under `workspace` —
        // proving the symlink was resolved rather than lexically kept.
        let canonical_outside = outside.canonicalize().unwrap();
        assert!(
            resolved.starts_with(&canonical_outside),
            "symlink parent should resolve to outside ({}), got {}",
            canonical_outside.display(),
            resolved.display()
        );
        assert_eq!(
            resolved.file_name(),
            Some(std::ffi::OsStr::new("new_file.txt"))
        );

        let _ = std::fs::remove_dir_all(&workspace);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[cfg(unix)]
    #[test]
    fn canonicalize_via_parent_resolves_symlink_through_missing_dirs() {
        // Multi-level escape: a symlink inside the workspace pointing outside,
        // traversed via a *not-yet-existing* intermediate directory
        // (`ws/escape/new_dir/file.txt`). Canonicalizing only the immediate
        // parent (`ws/escape/new_dir`) fails because `new_dir` is missing;
        // walking up to the existing symlink `ws/escape` resolves it, so the
        // path must land under `outside`, not lexically under the workspace.
        use super::canonicalize_via_parent;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."));
        let workspace = base.join(format!(".hi-symlink-ws2-{stamp}"));
        let outside = base.join(format!(".hi-symlink-out2-{stamp}"));
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let link = workspace.join("escape");
        std::os::unix::fs::symlink(&outside, &link).unwrap();

        // `new_dir` does not exist — the escape must still be detected.
        let target = link.join("new_dir").join("file.txt");
        let resolved = canonicalize_via_parent(&target);

        let canonical_outside = outside.canonicalize().unwrap();
        assert!(
            resolved.starts_with(&canonical_outside),
            "symlink should resolve to outside ({}) through the missing dir, got {}",
            canonical_outside.display(),
            resolved.display()
        );
        assert!(resolved.ends_with("new_dir/file.txt"));

        let _ = std::fs::remove_dir_all(&workspace);
        let _ = std::fs::remove_dir_all(&outside);
    }
}
