//! Path helpers (inlined to avoid a dependency on a separate paths crate).

use std::path::{Path, PathBuf};

/// Convert an absolute path to relative by stripping the root prefix.
///
/// Returns the path unchanged if not under `root`.
pub fn to_relative_path(root: &Path, abs_path: &Path) -> PathBuf {
    abs_path
        .strip_prefix(root)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|_| abs_path.to_path_buf())
}
