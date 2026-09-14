//! CLI parsing, config-file profiles, and resolution into effective settings.
//!
//! Precedence, highest first: explicit CLI flags → selected profile → env vars
//! → built-in defaults. Profiles let a user keep several models on hand
//! (e.g. a cloud Anthropic profile and a local Ollama profile) and use one with
//! `-p <name>`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use hi_ai::{CompatMode, DeepSeekCompat, OutputTokenParameter, ReasoningEffort, ToolMode};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_TOKENS: u32 = 8192;
const PIPENETWORK_DEFAULT_MAX_TOKENS: u32 = DEFAULT_MAX_TOKENS;
const LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS: u32 = 2048;

mod autoharnessfix;
mod cli;
mod credential_refs;
mod file;
mod harness;
mod mask;
mod profile_edit;
mod quality;
mod sentinel_flags;
mod session;
mod settings;
mod types;

#[cfg(test)]
mod credential_refs_tests;
#[cfg(test)]
mod sentinel_tests;
#[cfg(test)]
mod tests;

pub use autoharnessfix::*;
pub use cli::*;
pub(crate) use credential_refs::{migrate_api_key_env_to_literal, resolve_credential_reference};
pub use file::*;
pub use harness::*;
pub use mask::*;
pub use profile_edit::*;
pub use quality::*;
pub use sentinel_flags::*;
pub use session::*;
pub use settings::*;
pub use types::*;
