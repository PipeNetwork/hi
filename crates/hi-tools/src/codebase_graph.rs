//! Codebase-graph integration: a cached tree-sitter symbol index that backs
//! `definition`/`references` when LSP is unavailable.
//!
//! The [`IndexManager`] owns the index in a dedicated thread, keeps it warm
//! via file events, and answers go-to-definition / go-to-references queries
//! across Rust, TypeScript, JavaScript, Go, and Python. We spawn one manager
//! per workspace root (deduplicated internally by `IndexManager::spawn`).

use std::path::{Path, PathBuf};

use hi_codebase_graph::{IndexManager, IndexManagerConfig, IndexManagerHandle};

/// Spawn (or reuse) the per-workspace index manager. Returns `None` when the
/// index can't be started (e.g. no parseable files).
fn get_or_spawn(root: &Path) -> Option<std::sync::Arc<IndexManagerHandle>> {
    let config = IndexManagerConfig::new(root.to_path_buf());
    // IndexManager::spawn deduplicates per workspace per process internally.
    Some(IndexManager::spawn(config))
}

/// Run a go-to-definition or go-to-references query via the codebase graph.
///
/// Returns `None` when the index can't answer (send error or query error);
/// `Some(vec)` on success (possibly empty). Locations are formatted as
/// `path:line` (1-indexed, matching the `SymbolLocation` convention).
pub async fn query(
    root: &Path,
    file_path: &str,
    line: u32,
    column: u32,
    kind: &str,
) -> Option<Vec<String>> {
    let handle = get_or_spawn(root)?;
    let abs_path = if Path::new(file_path).is_absolute() {
        PathBuf::from(file_path)
    } else {
        root.join(file_path)
    };
    // The index uses 1-indexed rows/cols.
    let row = line as usize;
    let col = column as usize;
    let inner = if kind == "definition" {
        handle.goto_definition(abs_path, row, col).await.ok()?
    } else {
        handle
            .goto_references(abs_path, row, col, true)
            .await
            .ok()?
    };
    let query_result = inner.ok()?;
    Some(
        query_result
            .locations
            .iter()
            .map(|l| format!("{}:{}", l.path, l.line))
            .collect(),
    )
}
