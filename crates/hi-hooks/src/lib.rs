//! Runtime hook system for hi — file-based discovery, command execution,
//! and policy enforcement.
//!
//! Hooks are discovered from `.hi/hooks/` directories (project-local and
//! global `~/.hi/hooks/`), defined in TOML files, and executed as child
//! processes. Pre-tool hooks can block a tool call; post-tool hooks observe.
//!
//! # Quick start
//!
//! ```no_run
//! use hi_hooks::{HookRegistry, HookEvent, discover_hooks};
//! use std::path::Path;
//!
//! let (registry, errors) = discover_hooks(
//!     Some(Path::new("~/.hi/hooks")),
//!     Some(Path::new("./.hi/hooks")),
//! );
//! for err in &errors {
//!     eprintln!("hook load warning: {err}");
//! }
//! let pre_hooks = registry.hooks_for(HookEvent::PreToolUse);
//! println!("loaded {} pre_tool_use hooks", pre_hooks.len());
//! ```

mod config;
mod discovery;
mod event;
mod matcher;
mod result;
mod runner;

pub use config::{HandlerType, HookSpec};
pub use discovery::{HookRegistry, discover_hooks};
pub use event::{HookEvent, HookPayload, HookEventEnvelope};
pub use matcher::HookMatcher;
pub use result::{HookDecision, HookRunResult};
pub use runner::{run_hook, run_pre_tool_hooks, RunContext};
