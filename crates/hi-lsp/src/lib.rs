//! LSP client: spawn and speak the Language Server Protocol to external
//! language servers (rust-analyzer, pyright, etc.) over stdio JSON-RPC.
//!
//! The agent queries servers for diagnostics, definitions, references, and
//! hover — replacing grep-and-read for navigation and giving surgical
//! post-edit feedback. Servers are runtime deps discovered on `$PATH`; `hi`
//! does not vendor or re-implement them.
//!
//! See `LspManager` for the per-session entry point.

mod client;
mod detect;
mod manager;
mod protocol;

pub use detect::{Language, detect_language, server_command};
pub use manager::{LspManager, ServerStatus};
