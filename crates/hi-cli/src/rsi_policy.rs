//! Local trust-boundary policy for public RSI remote execution.
//!
//! This module owns **path safety and pack/context budgets** that the candidate
//! must not be allowed to relax. Transport, HTTP, and run lifecycle stay in
//! [`crate::rsi_remote`]; interactive managed-mode wiring stays in
//! [`crate::rsi_bootstrap`].
//!
//! See `docs/adr/001-rsi-runtime-boundary.md`.

use std::path::{Component, Path};

use anyhow::{Result, ensure};

/// Default RSI API host when config omits `base_url`.
pub(crate) const DEFAULT_BASE_URL: &str = "https://api.pipenetwork.ai";
/// Soft cap on gzipped workspace snapshot size.
pub(crate) const DEFAULT_COMPRESSED_BYTES: u64 = 256 * 1024 * 1024;
/// Soft cap on files packed into a snapshot.
pub(crate) const DEFAULT_ENTRIES: usize = 250_000;
/// Soft cap on conversation context uploaded with a run.
pub(crate) const DEFAULT_CONTEXT_BYTES: usize = 768 * 1024;
/// Hard cap on the canonical objective string.
pub(crate) const MAX_OBJECTIVE_BYTES: usize = 64 * 1024;
/// Managed-mode conversation reference budget (same order as default context).
pub(crate) const MANAGED_CONTEXT_BYTES: usize = 768 * 1024;

/// Snapshot size limits negotiated from server capabilities (with defaults).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotLimits {
    pub compressed_bytes: u64,
    pub entries: usize,
    pub uncompressed_bytes: u64,
}

/// Reject absolute paths, `..`, and non-UTF-8 components for RSI pack/apply.
pub(crate) fn validate_relative_path(path: &Path) -> Result<()> {
    ensure!(!path.is_absolute(), "RSI paths must be workspace-relative");
    ensure!(path.to_str().is_some(), "RSI paths must be UTF-8");
    for part in path.components() {
        ensure!(
            matches!(part, Component::Normal(_)),
            "unsafe RSI path: {}",
            path.display()
        );
    }
    Ok(())
}

/// Whether a relative path's first component is a reserved tree (`.git` / `.hi`).
pub(crate) fn is_reserved_workspace_root(path: &Path) -> bool {
    path.components().next().is_some_and(
        |part| matches!(part, Component::Normal(name) if name == ".git" || name == ".hi"),
    )
}

/// HTTPS (or loopback HTTP for tests) only.
pub(crate) fn validate_rsi_base_url(base_url: &str) -> Result<()> {
    ensure!(
        base_url.starts_with("https://")
            || base_url.starts_with("http://127.0.0.1")
            || base_url.starts_with("http://localhost"),
        "RSI base_url must use HTTPS (loopback HTTP is allowed for tests)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn rejects_parent_and_absolute_paths() {
        assert!(validate_relative_path(Path::new("../x")).is_err());
        assert!(validate_relative_path(Path::new("/etc/passwd")).is_err());
        assert!(validate_relative_path(Path::new("src/lib.rs")).is_ok());
    }

    #[test]
    fn reserved_roots() {
        assert!(is_reserved_workspace_root(Path::new(".git/config")));
        assert!(is_reserved_workspace_root(Path::new(".hi/memory.md")));
        assert!(!is_reserved_workspace_root(Path::new("src/main.rs")));
    }
}
