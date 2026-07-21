//! Hook configuration types — parsed from TOML hook files.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::event::HookEvent;
use crate::matcher::HookMatcher;

/// How a hook is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HandlerType {
    Command,
}

/// A fully parsed hook specification.
#[derive(Debug, Clone)]
pub struct HookSpec {
    /// Human-readable name for diagnostics.
    pub name: String,
    /// Which event triggers this hook.
    pub event: HookEvent,
    /// How the hook is executed.
    pub handler_type: HandlerType,
    /// The command to run (shell-expanded).
    pub command: String,
    /// Optional glob pattern to match tool names (for PreToolUse/PostToolUse).
    /// When `None`, matches all tools.
    pub matcher: Option<HookMatcher>,
    /// Timeout in seconds. `None` = no timeout.
    pub timeout_secs: Option<u64>,
    /// Whether the hook is enabled. Disabled hooks are loaded but skipped.
    pub enabled: bool,
    /// Source directory (for relative path resolution).
    pub source_dir: PathBuf,
    /// Extra environment variables for the hook process.
    pub extra_env: HashMap<String, String>,
}

/// Raw TOML structure for a hook file.
#[derive(Debug, Deserialize)]
pub struct HookFile {
    pub name: String,
    pub event: HookEvent,
    #[serde(default = "default_command")]
    pub handler_type: HandlerType,
    pub command: String,
    pub matcher: Option<String>,
    pub timeout: Option<u64>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

fn default_command() -> HandlerType {
    HandlerType::Command
}

fn default_true() -> bool {
    true
}

impl HookSpec {
    /// Build a `HookSpec` from a parsed `HookFile`, compiling the matcher.
    pub fn from_file(file: HookFile, source_dir: PathBuf) -> Result<Self, String> {
        let matcher = file
            .matcher
            .map(|pattern| HookMatcher::new(&pattern))
            .transpose()
            .map_err(|e| format!("invalid matcher pattern: {e}"))?;
        Ok(Self {
            name: file.name,
            event: file.event,
            handler_type: file.handler_type,
            command: file.command,
            matcher,
            timeout_secs: file.timeout,
            enabled: file.enabled,
            source_dir,
            extra_env: file.env,
        })
    }
}
