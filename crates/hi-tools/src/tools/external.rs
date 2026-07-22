//! External tool handlers: MCP, memory, and skill invocation.
//!
//! These tools extend the agent beyond the built-in file/shell/web toolkit:
//!
//! - `use_tool` / `search_tool` — call external MCP (Model Context Protocol)
//!   tool servers. The MCP client connection is supplied by the frontend via
//!   a trait, so the agent crate doesn't depend on any specific MCP transport.
//! - `memory_search` / `memory_get` — cross-session knowledge retrieval from
//!   an indexed memory store. Also frontend-supplied via a trait.
//! - `skill` — invoke a named learned skill. Skills are indexed from project
//!   or user config; the handler returns the skill's procedure text.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ToolOutcome;

// ───────────────────────────────────────────────────────────────────────────
// MCP — Model Context Protocol tool integration
// ───────────────────────────────────────────────────────────────────────────

/// Description of a single MCP tool discovered via `search_tool`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpToolInfo {
    /// MCP server name.
    pub server: String,
    /// Tool name on that server.
    pub tool: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// JSON Schema for the tool's parameters.
    pub schema: Value,
}

/// Trait for MCP tool discovery and invocation.
///
/// Implemented by the frontend (hi-cli) which owns the MCP client connections.
/// The agent crate depends only on this trait, not on any specific MCP transport.
#[async_trait::async_trait]
pub trait McpBackend: Send + Sync {
    /// List available tools across all connected MCP servers, optionally
    /// filtered by a query string (matches tool name or description).
    async fn search(&self, query: Option<&str>) -> Result<Vec<McpToolInfo>>;

    /// Call a tool on a specific MCP server with the given JSON arguments.
    async fn call(&self, server: &str, tool: &str, arguments: &Value) -> Result<String>;
}

/// Run the `search_tool` tool — discover available MCP tools.
pub async fn run_search_tool(
    backend: Option<&dyn McpBackend>,
    arguments: &str,
) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct Args {
        #[serde(default)]
        query: Option<String>,
    }
    let args: Args = serde_json::from_str(arguments).context("invalid tool arguments")?;

    let backend = match backend {
        Some(b) => b,
        None => {
            return Ok(ToolOutcome::plain(
                "No MCP servers are connected. Use the frontend's MCP configuration to connect external tool servers.".to_string(),
            ));
        }
    };

    let tools = backend.search(args.query.as_deref()).await?;

    if tools.is_empty() {
        return Ok(ToolOutcome::plain(
            "No MCP tools found. Connect MCP servers via the frontend configuration.".to_string(),
        ));
    }

    let mut lines = Vec::with_capacity(tools.len());
    for info in &tools {
        lines.push(format!(
            "  {} / {} — {}",
            info.server, info.tool, info.description
        ));
    }

    let content = format!(
        "Available MCP tools ({}):\n{}",
        tools.len(),
        lines.join("\n")
    );

    Ok(ToolOutcome::plain(content))
}

/// Run the `use_tool` tool — call an external MCP tool.
pub async fn run_use_tool(
    backend: Option<&dyn McpBackend>,
    arguments: &str,
) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct Args {
        server: String,
        tool: String,
        #[serde(default)]
        arguments: Value,
    }
    let args: Args = serde_json::from_str(arguments).context("invalid tool arguments")?;

    let backend = match backend {
        Some(b) => b,
        None => {
            return Ok(ToolOutcome::failed(
                "No MCP servers are connected. Use `search_tool` to discover available tools after connecting an MCP server.".to_string(),
            ));
        }
    };

    let result = backend
        .call(&args.server, &args.tool, &args.arguments)
        .await?;

    Ok(ToolOutcome::plain(result))
}

// ───────────────────────────────────────────────────────────────────────────
// Memory — cross-session knowledge retrieval
// ───────────────────────────────────────────────────────────────────────────

/// A single memory search result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemorySearchResult {
    /// Memory entry path (for use with `memory_get`).
    pub path: String,
    /// Relevance score (higher = more relevant).
    pub score: f64,
    /// Text snippet of the matching content.
    pub snippet: String,
}

/// Trait for cross-session memory retrieval.
///
/// Implemented by the frontend which owns the memory index/store.
#[async_trait::async_trait]
pub trait MemoryBackend: Send + Sync {
    /// Search indexed memory for chunks relevant to `query`.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemorySearchResult>>;

    /// Read a specific memory entry by its path.
    async fn get(&self, path: &str) -> Result<String>;
}

/// Run the `memory_search` tool.
pub async fn run_memory_search(
    backend: Option<&dyn MemoryBackend>,
    arguments: &str,
) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct Args {
        query: String,
        #[serde(default = "default_limit")]
        limit: usize,
    }
    fn default_limit() -> usize {
        5
    }
    let args: Args = serde_json::from_str(arguments).context("invalid tool arguments")?;

    let backend = match backend {
        Some(b) => b,
        None => {
            return Ok(ToolOutcome::plain(
                "No memory store is configured. Memory tools require a memory backend to be set up by the frontend.".to_string(),
            ));
        }
    };

    let results = backend.search(&args.query, args.limit).await?;

    if results.is_empty() {
        return Ok(ToolOutcome::plain(
            "No memory entries found for this query.".to_string(),
        ));
    }

    let mut lines = Vec::with_capacity(results.len());
    for result in &results {
        lines.push(format!(
            "[{}] (score: {:.2}) {}\n  path: {}",
            result.snippet, result.score, result.snippet, result.path
        ));
    }

    Ok(ToolOutcome::plain(format!(
        "Found {} memory entries:\n\n{}",
        results.len(),
        lines.join("\n\n")
    )))
}

/// Run the `memory_get` tool.
pub async fn run_memory_get(
    backend: Option<&dyn MemoryBackend>,
    arguments: &str,
) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct Args {
        path: String,
    }
    let args: Args = serde_json::from_str(arguments).context("invalid tool arguments")?;

    let backend = match backend {
        Some(b) => b,
        None => {
            return Ok(ToolOutcome::plain(
                "No memory store is configured.".to_string(),
            ));
        }
    };

    let content = backend.get(&args.path).await?;

    Ok(ToolOutcome::plain(content))
}

// ───────────────────────────────────────────────────────────────────────────
// Skill — named learned skill invocation
// ───────────────────────────────────────────────────────────────────────────

/// Trait for skill lookup.
///
/// Implemented by the frontend which indexes skills from project/user config.
pub trait SkillBackend: Send + Sync {
    /// Look up a skill by name. Returns the skill's procedure text, or None
    /// if no skill with that name exists.
    fn lookup(&self, name: &str) -> Option<String>;

    /// List all available skill names.
    fn list(&self) -> Vec<String>;
}

/// Run the `skill` tool — invoke a named learned skill.
pub fn run_skill(backend: Option<&dyn SkillBackend>, arguments: &str) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct Args {
        name: String,
        #[serde(default)]
        args: Option<String>,
    }
    let args: Args = serde_json::from_str(arguments).context("invalid tool arguments")?;

    let backend = match backend {
        Some(b) => b,
        None => {
            return Ok(ToolOutcome::plain(
                "No skill registry is configured. Skills are indexed from project and user config by the frontend.".to_string(),
            ));
        }
    };

    match backend.lookup(&args.name) {
        Some(procedure) => {
            let content = if let Some(extra) = args.args {
                format!(
                    "Skill: {}\nArguments: {}\n\n{}",
                    args.name, extra, procedure
                )
            } else {
                format!("Skill: {}\n\n{}", args.name, procedure)
            };
            Ok(ToolOutcome::plain(content))
        }
        None => {
            let available = backend.list();
            if available.is_empty() {
                bail!(
                    "no skill named '{}' exists, and no skills are registered",
                    args.name
                );
            }
            bail!(
                "no skill named '{}' exists. Available skills: {}",
                args.name,
                available.join(", ")
            );
        }
    }
}
