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
    // For paths that exist, canonicalize to resolve symlinks and `..`.
    let canonical = resolved.canonicalize().unwrap_or(resolved.clone());
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

/// VCS metadata directories that must never reach the model. We walk with
/// `hidden(false)` so the agent can see useful dotfiles (`.github/`,
/// `.env.example`, `.cargo/config.toml`, …), but these internal directories are
/// large, mostly binary, and leak repository internals (loose/packed objects,
/// refs, reflogs, config). Used as a `WalkBuilder::filter_entry` predicate,
/// which prunes the whole subtree so we never even descend into them.
pub(crate) fn is_vcs_metadata_dir(entry: &ignore::DirEntry) -> bool {
    matches!(
        entry.file_name().to_str(),
        Some(".git" | ".hg" | ".svn" | ".jj")
    )
}

/// Maximum number of cached file reads. Beyond this, the cache is cleared
/// entirely (cheap — it refills lazily on the next re-read).
pub(crate) const READ_CACHE_MAX: usize = 50;

/// Per-turn cache of file reads, so re-reading the same file (common when the
/// model is orienting) hits memory instead of disk. Cleared between turns, and
/// bounded to [`READ_CACHE_MAX`] entries to avoid unbounded memory growth.
pub(crate) static READ_CACHE: LazyLock<Mutex<HashMap<String, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Clear the per-turn read cache. Call at the start of each turn.
pub fn clear_read_cache() {
    if let Ok(mut cache) = READ_CACHE.lock() {
        cache.clear();
    }
}
