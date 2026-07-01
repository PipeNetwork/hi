//! Process-global LSP handle so the tool layer can reach the `LspManager`
//! without threading it through every `execute` call site (mirroring how
//! `READ_CACHE` works in `paths.rs`).

use std::path::Path;
use std::sync::{Arc, OnceLock};

use hi_lsp::{LspManager, ServerStatus};

static LSP: OnceLock<Arc<LspManager>> = OnceLock::new();

/// Install the process-global LSP manager. Called once at session start by
/// the agent. Subsequent calls are no-ops (the first wins).
pub fn set_lsp_manager(mgr: LspManager) {
    let _ = LSP.set(Arc::new(mgr));
}

/// Install an already-`Arc`ed manager (for when the caller needs to keep a
/// handle to toggle enabled state).
pub fn set_lsp_manager_arc(mgr: Arc<LspManager>) {
    let _ = LSP.set(mgr);
}

/// Whether LSP is enabled for this session.
pub async fn lsp_enabled() -> bool {
    match LSP.get() {
        Some(mgr) => mgr.is_enabled().await,
        None => false,
    }
}

/// Status of each known language server, for `/lsp status`.
pub async fn lsp_status() -> Vec<ServerStatus> {
    match LSP.get() {
        Some(mgr) => mgr.status().await,
        None => Vec::new(),
    }
}

/// Sync status (no async runtime needed), for command handlers.
pub fn lsp_status_sync() -> Vec<ServerStatus> {
    match LSP.get() {
        Some(mgr) => mgr.status_sync(),
        None => Vec::new(),
    }
}

/// A human-readable, frontend-agnostic status summary for `/lsp` (no arg)
/// and `/lsp status`. Shows enabled state plus per-language availability and
/// running state, so users can see what's actually wired up at a glance.
pub fn lsp_status_report(enabled: bool) -> String {
    let servers = lsp_status_sync();
    let mut out = format!("LSP: {}\n", if enabled { "on" } else { "off" });
    if servers.is_empty() {
        out.push_str("  (no language servers configured)");
        return out;
    }
    for s in &servers {
        let lang = match s.language {
            hi_lsp::Language::Rust => "rust",
            hi_lsp::Language::Python => "python",
            hi_lsp::Language::Go => "go",
            hi_lsp::Language::TypeScript => "typescript",
        };
        let state = if !s.available {
            "not installed"
        } else if s.running {
            "running"
        } else {
            "available"
        };
        out.push_str(&format!("  {lang:<12} {state}\n"));
    }
    out.trim_end().to_string()
}

/// Push updated file contents to the LSP server after an edit. No-op when LSP
/// is off or no manager is installed.
pub async fn sync_lsp_document(path: &Path, text: &str) {
    if let Some(mgr) = LSP.get() {
        let _ = mgr.sync_document(path, text).await;
    }
}

/// Access the manager for direct tool queries (diagnostics, definition, etc.).
pub(crate) fn lsp_manager() -> Option<Arc<LspManager>> {
    LSP.get().cloned()
}

/// Public accessor for the agent to toggle enabled state.
pub fn lsp_manager_handle() -> Option<Arc<LspManager>> {
    LSP.get().cloned()
}
