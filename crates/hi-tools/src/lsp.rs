//! Frontend-agnostic formatting for workspace-owned LSP status.

use hi_lsp::ServerStatus;

/// Format status from an explicit workspace-local manager.
pub fn lsp_status_report_for(enabled: bool, servers: &[ServerStatus]) -> String {
    let mut out = format!("LSP: {}\n", if enabled { "on" } else { "off" });
    if servers.is_empty() {
        out.push_str("  (no language servers configured)");
        return out;
    }
    for status in servers {
        let language = match status.language {
            hi_lsp::Language::Rust => "rust",
            hi_lsp::Language::Python => "python",
            hi_lsp::Language::Go => "go",
            hi_lsp::Language::TypeScript => "typescript",
        };
        let state = if !status.available {
            "not installed"
        } else if status.running {
            "running"
        } else {
            "available"
        };
        out.push_str(&format!("  {language:<12} {state}\n"));
    }
    out.trim_end().to_string()
}
