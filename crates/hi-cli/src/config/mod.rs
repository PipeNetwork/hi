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
use hi_agent::{LspMode, ReviewPolicy, ToolSet, VerificationMode, VerifyStage};
use hi_ai::{CompatMode, ReasoningEffort, ToolMode};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_TOKENS: u32 = 8192;
const PIPENETWORK_DEFAULT_MAX_TOKENS: u32 = DEFAULT_MAX_TOKENS;
const LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS: u32 = 2048;

mod cli;
mod file;
mod profile_edit;
mod quality;
mod session;
mod settings;

#[cfg(test)]
mod tests;

pub use cli::*;
pub use file::*;
pub use profile_edit::*;
pub use quality::*;
pub use session::*;
pub use settings::*;
