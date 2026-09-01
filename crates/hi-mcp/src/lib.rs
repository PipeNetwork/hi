//! General MCP (Model Context Protocol) host framework for `hi`.
//!
//! Provides a client for connecting to MCP servers (stdio or HTTP transport),
//! discovering their tools, and invoking them. MCP is an open protocol that
//! lets AI assistants connect to external tools and data sources.
//!
//! This crate defines the core types and traits for MCP integration:
//! - [`McpServer`] — a configured MCP server connection
//! - [`McpTransport`] — the communication transport (stdio, HTTP)
//! - [`McpTool`] — a tool discovered from an MCP server
//! - [`McpClient`] — the client that manages server connections and tool calls
//!
//! Inspired by grok-build's `xai-grok-mcp` crate.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn run() -> anyhow::Result<()> {
//! use hi_mcp::{McpClient, McpServerConfig};
//!
//! let mut client = McpClient::new();
//! let server = McpServerConfig::stdio("my-server", "npx", &["-y", "@modelcontextprotocol/server-sqlite"]);
//! client.connect(server).await?;
//! let tools = client.list_tools("my-server")?;
//! for tool in tools {
//!     println!("{}: {}", tool.name, tool.description);
//! }
//! # Ok(())
//! # }
//! ```

mod allowlist;
mod discover;
mod http;
mod serve;

pub use allowlist::{
    AgentToolPolicy, PIPE_DEFAULT_ALLOWED_TOOLS, PIPE_NEVER_AGENT_TOOLS, PIPE_SERVER_NAME,
    ServerAllowList, is_never_agent_pipe_tool,
};
pub use discover::{
    DiscoveredMcpServer, McpConfigSource, McpDiscoveryPaths, McpImportPolicy, McpSourceFilter,
    blocked_names, default_codex_config_path, default_discovery_paths, discover_all_servers,
    discover_all_servers_from, expand_env_templates, expand_http_headers, load_claude_mcp_json,
    load_codex_servers, parse_claude_mcp_json, parse_codex_mcp_toml, transport_label,
};
pub use serve::{
    McpStdioHandler, dispatch_line, handle_message, hi_serve_tools, serve_stdio, serve_stdio_io,
};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use thiserror::Error;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

/// Errors from the MCP client.
#[derive(Debug, Error)]
pub enum McpError {
    /// The server is not connected.
    #[error("server not connected: {0}")]
    NotConnected(String),
    /// The server was not found.
    #[error("server not found: {0}")]
    ServerNotFound(String),
    /// The transport failed.
    #[error("transport error: {0}")]
    Transport(String),
    /// The server returned an error.
    #[error("server error: {0}")]
    Server(String),
    /// Tool invocation failed.
    #[error("tool invocation failed: {0}")]
    ToolInvocation(String),
    /// Authentication failed.
    #[error("auth error: {0}")]
    Auth(String),
    /// An I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization/deserialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// The server returned a response for a different request id.
    #[error("json-rpc response id mismatch: expected {expected}, got {actual}")]
    ResponseIdMismatch { expected: u64, actual: String },
}

/// The transport for an MCP server connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpTransport {
    /// Communicate over stdio (spawn a child process).
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    },
    /// Communicate over HTTP (connect to a URL).
    Http {
        url: String,
        headers: HashMap<String, String>,
    },
}

/// Configuration for an MCP server connection.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// A unique name for this server connection.
    pub name: String,
    /// The transport to use.
    pub transport: McpTransport,
    /// Whether to auto-reconnect on failure.
    pub auto_reconnect: bool,
    /// When non-empty, only these tools are agent-visible.
    pub only: Vec<String>,
    /// Tools hidden from the agent. Wins over [`Self::only`].
    pub exclude: Vec<String>,
}

impl McpServerConfig {
    /// Create a stdio server config.
    pub fn stdio(name: impl Into<String>, command: impl Into<String>, args: &[&str]) -> Self {
        Self {
            name: name.into(),
            transport: McpTransport::Stdio {
                command: command.into(),
                args: args.iter().map(|s| s.to_string()).collect(),
                env: HashMap::new(),
            },
            auto_reconnect: true,
            only: Vec::new(),
            exclude: Vec::new(),
        }
    }

    /// Create an HTTP server config.
    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transport: McpTransport::Http {
                url: url.into(),
                headers: HashMap::new(),
            },
            auto_reconnect: true,
            only: Vec::new(),
            exclude: Vec::new(),
        }
    }

    /// Set an environment variable (stdio transport only).
    #[must_use]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let McpTransport::Stdio { env, .. } = &mut self.transport {
            env.insert(key.into(), value.into());
        }
        self
    }

    /// Set a header (HTTP transport only).
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let McpTransport::Http { headers, .. } = &mut self.transport {
            headers.insert(key.into(), value.into());
        }
        self
    }
}

/// A tool discovered from an MCP server.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct McpTool {
    /// The tool name (unique within a server).
    pub name: String,
    /// A human-readable description of what the tool does.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    #[serde(default, rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

impl McpTool {
    /// Convert an MCP tool into hi's shared admission/host descriptor.
    pub fn descriptor(&self, server_name: &str) -> hi_tools::descriptors::ToolDescriptor {
        hi_tools::descriptors::ToolDescriptor {
            name: format!("mcp::{server_name}::{}", self.name),
            input_schema: self.input_schema.clone(),
            output_schema: serde_json::json!({}),
            required_capabilities: [format!("mcp:{server_name}")].into_iter().collect(),
            side_effect: hi_tools::descriptors::SideEffect::Process,
            maximum_output_bytes: 2 * 1024 * 1024,
            timeout_ms: 120_000,
            replayable: false,
        }
    }
}

/// The result of invoking an MCP tool.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct McpToolResult {
    /// The text content returned by the tool.
    pub content: String,
    /// Whether the tool invocation resulted in an error.
    pub is_error: bool,
}

/// A resource discovered from an MCP server.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct McpResource {
    /// The resource URI.
    pub uri: String,
    /// A human-readable name.
    pub name: String,
    /// A description of the resource.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The MIME type of the resource.
    #[serde(default, rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Liveness state of an MCP server connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerStatus {
    /// The server is connected and responding.
    Connected,
    /// The server is disconnected.
    Disconnected,
    /// The server failed to connect or has crashed.
    Failed,
    /// The server is connecting.
    Connecting,
    /// Registered but blocked by import policy (visible in `/mcp`, not callable).
    Disabled,
}

/// A connected (or registered) MCP server.
pub struct McpServer {
    /// The server configuration.
    pub config: McpServerConfig,
    /// The current status.
    pub status: ServerStatus,
    /// The server's reported name (from the initialize handshake).
    pub server_name: Option<String>,
    /// The server's reported version.
    pub server_version: Option<String>,
    /// Discovered tools.
    pub tools: Vec<McpTool>,
    /// Discovered resources.
    pub resources: Vec<McpResource>,
    /// Config source (`.hi/mcp`, Claude, Codex).
    pub source: McpConfigSource,
    /// Whether the server may be connected.
    pub enabled: bool,
    /// Why enable is refused (import policy). Sticky — [`McpClient::set_enabled`]
    /// cannot silently clear this.
    pub blocked_reason: Option<String>,
    /// Last connect/call error, if any.
    pub last_error: Option<String>,
    generation: u64,
    transport: Option<Box<dyn McpTransportTrait>>,
}

/// A line-delimited JSON-RPC transport for MCP servers launched as child
/// processes. MCP stdio servers must keep stdout protocol-clean; stderr is
/// intentionally discarded here because it is diagnostic, not protocol data.
struct StdioTransport {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl StdioTransport {
    async fn spawn(
        config: &McpServerConfig,
        process_runner: Option<&hi_tools::ProcessRunner>,
    ) -> Result<Self, McpError> {
        let McpTransport::Stdio { command, args, env } = &config.transport else {
            return Err(McpError::Transport("stdio transport required".into()));
        };
        if command.trim().is_empty() {
            return Err(McpError::Transport("stdio MCP command is empty".into()));
        }
        let mut child = if let Some(runner) = process_runner {
            runner
                .spawn_program_piped(command, args, env)
                .map_err(|error| {
                    McpError::Transport(format!(
                        "failed to spawn MCP server '{command}': {error:#}"
                    ))
                })?
        } else {
            let mut process = Command::new(command);
            process
                .args(args)
                .envs(env)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .kill_on_drop(true);
            process.spawn().map_err(|error| {
                McpError::Transport(format!("failed to spawn MCP server '{command}': {error}"))
            })?
        };
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("MCP server stdin was not piped".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("MCP server stdout was not piped".into()))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        })
    }

    async fn write_message(&mut self, value: &serde_json::Value) -> Result<(), McpError> {
        let mut encoded = serde_json::to_vec(value)?;
        encoded.push(b'\n');
        self.stdin.write_all(&encoded).await?;
        self.stdin.flush().await?;
        Ok(())
    }
}

#[async_trait]
impl McpTransportTrait for StdioTransport {
    async fn request(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let mut request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
        });
        if let Some(params) = params {
            request["params"] = params;
        }
        self.write_message(&request).await?;
        while let Some(line) = next_rpc_line(&mut self.stdout).await? {
            if line.trim().is_empty() {
                continue;
            }
            let message: serde_json::Value = serde_json::from_str(&line)?;
            // Servers may emit notifications while a request is in flight.
            let Some(response_id) = message.get("id") else {
                continue;
            };
            if response_id.is_null() {
                continue;
            }
            if response_id.as_u64() != Some(id) {
                return Err(McpError::ResponseIdMismatch {
                    expected: id,
                    actual: response_id.to_string(),
                });
            }
            if let Some(error) = message.get("error") {
                return Err(McpError::Server(error.to_string()));
            }
            return Ok(message
                .get("result")
                .cloned()
                .unwrap_or(serde_json::Value::Null));
        }
        let status = self.child.try_wait()?.map(|status| status.to_string());
        Err(McpError::Transport(format!(
            "MCP server closed stdout{}",
            status.map(|s| format!(" ({s})")).unwrap_or_default()
        )))
    }

    async fn notify(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), McpError> {
        let mut notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(params) = params {
            notification["params"] = params;
        }
        self.write_message(&notification).await
    }

    async fn close(&mut self) -> Result<(), McpError> {
        self.stdin.shutdown().await?;
        let _ = self.child.kill().await;
        Ok(())
    }
}

/// Trait for MCP transports. Implementations handle the actual communication.
#[async_trait]
pub trait McpTransportTrait: Send + Sync {
    /// Send a JSON-RPC request and receive a response.
    async fn request(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError>;

    /// Send a JSON-RPC notification (no response expected).
    async fn notify(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), McpError>;

    /// Close the transport.
    async fn close(&mut self) -> Result<(), McpError>;
}

/// Protocol version advertised on `initialize`. Pipe MCP speaks 2025-06-18
/// and echoes the client's request; older stdio servers typically accept it.
pub const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

/// The MCP client that manages server connections and tool calls.
pub struct McpClient {
    servers: HashMap<String, McpServer>,
    process_runner: Option<std::sync::Arc<hi_tools::ProcessRunner>>,
    policy: AgentToolPolicy,
    workspace_root: Option<PathBuf>,
}

impl McpClient {
    /// Create a new MCP client with no connected servers.
    ///
    /// The current directory is used as the default workspace root so stdio
    /// children go through hi's process/sandbox boundary. Call
    /// [`Self::with_process_runner`] when the caller has a more precise root.
    pub fn new() -> Self {
        let process_runner = std::env::current_dir()
            .ok()
            .and_then(|root| hi_tools::ProcessRunner::new(root).ok())
            .map(std::sync::Arc::new);
        Self {
            servers: HashMap::new(),
            process_runner,
            policy: AgentToolPolicy::new(),
            workspace_root: std::env::current_dir().ok(),
        }
    }

    /// Use hi's sanitized process/sandbox boundary for future stdio servers.
    pub fn with_process_runner(runner: hi_tools::ProcessRunner) -> Self {
        let workspace_root = Some(runner.root().to_path_buf());
        Self {
            servers: HashMap::new(),
            process_runner: Some(std::sync::Arc::new(runner)),
            policy: AgentToolPolicy::new(),
            workspace_root,
        }
    }

    pub fn workspace_root(&self) -> Option<&Path> {
        self.workspace_root.as_deref()
    }

    pub fn policy(&self) -> &AgentToolPolicy {
        &self.policy
    }

    /// Overlay `[mcp.servers.<name>]` lists from hi.toml.
    pub fn overlay_server_policies<'a>(
        &mut self,
        policies: impl IntoIterator<Item = (&'a str, &'a ServerAllowList)>,
    ) {
        for (name, list) in policies {
            self.policy.overlay(name, list);
        }
    }

    /// Replace the agent-facing tool policy (first-party Pipe allowlist).
    pub fn set_agent_tool_policy(&mut self, policy: AgentToolPolicy) {
        self.policy = policy;
    }

    /// Register a discovered server without connecting. Startup stays fail-open.
    pub fn register(&mut self, discovered: DiscoveredMcpServer) {
        let name = discovered.config.name.clone();
        let status = if discovered.enabled {
            ServerStatus::Disconnected
        } else {
            ServerStatus::Disabled
        };
        self.policy.set_from_config(
            &discovered.config.name,
            &discovered.config.only,
            &discovered.config.exclude,
        );
        self.servers.insert(
            name,
            McpServer {
                config: discovered.config,
                status,
                server_name: None,
                server_version: None,
                tools: Vec::new(),
                resources: Vec::new(),
                source: discovered.source,
                enabled: discovered.enabled,
                blocked_reason: discovered.blocked_reason,
                last_error: None,
                generation: 0,
                transport: None,
            },
        );
    }

    /// Register many servers (including blocked ones).
    pub fn register_all(&mut self, discovered: impl IntoIterator<Item = DiscoveredMcpServer>) {
        for item in discovered {
            self.register(item);
        }
    }

    /// Connect to an MCP server immediately (eager). Prefer [`Self::register`] +
    /// [`Self::ensure_connected`] for workspace startup.
    pub async fn connect(&mut self, config: McpServerConfig) -> Result<(), McpError> {
        let name = config.name.clone();
        self.register(DiscoveredMcpServer {
            config,
            source: McpConfigSource::Hi,
            enabled: true,
            blocked_reason: None,
        });
        self.ensure_connected(&name, Duration::from_secs(30)).await
    }

    /// Wait up to `grace` for a registered server to handshake. Fail-fast with
    /// an actionable `/mcp <name> reconnect` hint.
    pub async fn ensure_connected(&mut self, name: &str, grace: Duration) -> Result<(), McpError> {
        match tokio::time::timeout(grace, self.connect_inner(name)).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(err),
            Err(_) => {
                if let Some(server) = self.servers.get_mut(name) {
                    server.status = ServerStatus::Failed;
                    server.last_error = Some("connect timed out".into());
                }
                Err(McpError::NotConnected(format!(
                    "server '{name}' did not connect in time — try `/mcp {name} reconnect`"
                )))
            }
        }
    }

    /// Force a new handshake (bumps generation so a stale close cannot clobber it).
    pub async fn reconnect(&mut self, name: &str) -> Result<(), McpError> {
        self.disconnect_transport(name).await;
        self.ensure_connected(name, Duration::from_secs(8)).await
    }

    /// Enable or disable a registered server. Blocked import names cannot be
    /// enabled.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> Result<(), McpError> {
        let server = self
            .servers
            .get_mut(name)
            .ok_or_else(|| McpError::ServerNotFound(name.to_string()))?;
        if enabled {
            if let Some(reason) = &server.blocked_reason {
                return Err(McpError::Server(format!(
                    "cannot enable '{name}': {reason}"
                )));
            }
            server.enabled = true;
            if server.status == ServerStatus::Disabled {
                server.status = ServerStatus::Disconnected;
            }
        } else {
            server.enabled = false;
            server.status = ServerStatus::Disabled;
        }
        Ok(())
    }

    async fn spawn_transport(
        &self,
        config: &McpServerConfig,
    ) -> Result<Box<dyn McpTransportTrait>, McpError> {
        match &config.transport {
            McpTransport::Stdio { .. } => {
                let t = StdioTransport::spawn(config, self.process_runner.as_deref()).await?;
                Ok(Box::new(t) as Box<dyn McpTransportTrait>)
            }
            McpTransport::Http { url, headers } => {
                let t = http::HttpTransport::connect(url.clone(), headers.clone())?;
                Ok(Box::new(t) as Box<dyn McpTransportTrait>)
            }
        }
    }

    async fn connect_inner(&mut self, name: &str) -> Result<(), McpError> {
        let config = {
            let server = self
                .servers
                .get_mut(name)
                .ok_or_else(|| McpError::ServerNotFound(name.to_string()))?;
            if !server.enabled {
                return Err(McpError::Server(format!(
                    "server '{name}' is disabled{}",
                    server
                        .blocked_reason
                        .as_deref()
                        .map(|r| format!(" ({r})"))
                        .unwrap_or_default()
                )));
            }
            if server.status == ServerStatus::Connected && server.transport.is_some() {
                return Ok(());
            }
            server.generation = server.generation.saturating_add(1);
            server.status = ServerStatus::Connecting;
            server.config.clone()
        };
        let generation = self.servers.get(name).map(|s| s.generation).unwrap_or(0);
        let handshake = async {
            let mut transport = self.spawn_transport(&config).await?;
            let initialize = transport
                .request(
                    "initialize",
                    Some(serde_json::json!({
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "hi", "version": env!("CARGO_PKG_VERSION")}
                    })),
                )
                .await?;
            // Pipe (and some other servers) do not implement this notification.
            let _ = transport.notify("notifications/initialized", None).await;
            let capabilities = initialize.get("capabilities");
            let tools = if capabilities
                .and_then(|capabilities| capabilities.get("tools"))
                .is_some()
            {
                list_tools_from_result(transport.request("tools/list", None).await?)?
            } else {
                Vec::new()
            };
            let resources = if capabilities
                .and_then(|capabilities| capabilities.get("resources"))
                .is_some()
            {
                list_resources_from_result(transport.request("resources/list", None).await?)?
            } else {
                Vec::new()
            };
            Ok::<_, McpError>((initialize, tools, resources, transport))
        }
        .await;
        let stale = self
            .servers
            .get(name)
            .is_none_or(|server| server.generation != generation);
        if stale {
            if let Ok((_, _, _, mut transport)) = handshake {
                let _ = transport.close().await;
            }
            return Err(McpError::Transport(format!(
                "stale connect for '{name}' discarded (generation changed)"
            )));
        }
        let server = self
            .servers
            .get_mut(name)
            .ok_or_else(|| McpError::ServerNotFound(name.to_string()))?;
        match handshake {
            Ok((initialize, tools, resources, transport)) => {
                server.status = ServerStatus::Connected;
                server.last_error = None;
                server.server_name = initialize
                    .get("serverInfo")
                    .and_then(|value| value.get("name"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
                server.server_version = initialize
                    .get("serverInfo")
                    .and_then(|value| value.get("version"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
                server.tools = tools;
                server.resources = resources;
                server.transport = Some(transport);
                Ok(())
            }
            Err(err) => {
                server.status = ServerStatus::Failed;
                server.last_error = Some(err.to_string());
                server.transport = None;
                Err(err)
            }
        }
    }

    async fn disconnect_transport(&mut self, name: &str) {
        if let Some(server) = self.servers.get_mut(name)
            && let Some(mut transport) = server.transport.take()
        {
            server.generation = server.generation.saturating_add(1);
            let _ = transport.close().await;
            if server.enabled {
                server.status = ServerStatus::Disconnected;
            }
        }
    }

    /// Disconnect from a server.
    pub async fn disconnect(&mut self, name: &str) -> Result<(), McpError> {
        let mut server = self
            .servers
            .remove(name)
            .ok_or_else(|| McpError::ServerNotFound(name.to_string()))?;
        if let Some(mut transport) = server.transport.take() {
            transport.close().await?;
        }
        Ok(())
    }

    /// List connected server names.
    pub fn server_names(&self) -> Vec<String> {
        self.servers.keys().cloned().collect()
    }

    /// Get the status of a server.
    pub fn status(&self, name: &str) -> Result<ServerStatus, McpError> {
        self.servers
            .get(name)
            .map(|s| s.status)
            .ok_or_else(|| McpError::ServerNotFound(name.to_string()))
    }

    /// List **agent-visible** tools from a server (allowlist applied).
    pub fn list_tools(&self, name: &str) -> Result<Vec<McpTool>, McpError> {
        let server = self
            .servers
            .get(name)
            .ok_or_else(|| McpError::ServerNotFound(name.to_string()))?;
        Ok(server
            .tools
            .iter()
            .filter(|tool| self.policy.allows(server.source, name, &tool.name))
            .cloned()
            .collect())
    }

    pub fn tool_descriptors(
        &self,
        name: &str,
    ) -> Result<Vec<hi_tools::descriptors::ToolDescriptor>, McpError> {
        Ok(self
            .list_tools(name)?
            .iter()
            .map(|tool| tool.descriptor(name))
            .collect())
    }

    /// List resources from a server.
    pub fn list_resources(&self, name: &str) -> Result<&[McpResource], McpError> {
        self.servers
            .get(name)
            .map(|s| s.resources.as_slice())
            .ok_or_else(|| McpError::ServerNotFound(name.to_string()))
    }

    pub const LAZY_CONNECT_GRACE: Duration = Duration::from_secs(2);

    /// Workspace `/mcp` status table.
    pub fn status_table(&self) -> String {
        if self.servers.is_empty() {
            return "workspace MCP: (none registered)\n  configure `.hi/mcp/*.json` or `.mcp.json`\n  `/mcp pipe` inspects the provider mcp_url\n".into();
        }
        let mut out = String::from("workspace MCP:\n");
        out.push_str("  name            source  transport  status        tools\n");
        let mut names: Vec<_> = self.servers.keys().cloned().collect();
        names.sort();
        for name in names {
            let Some(server) = self.servers.get(&name) else {
                continue;
            };
            let status = match server.status {
                ServerStatus::Connected => "connected",
                ServerStatus::Disconnected => "registered",
                ServerStatus::Connecting => "connecting",
                ServerStatus::Failed => "failed",
                ServerStatus::Disabled => "disabled",
            };
            let extra = server
                .blocked_reason
                .as_deref()
                .or(server.last_error.as_deref())
                .unwrap_or("");
            out.push_str(&format!(
                "  {:<15} {:<7} {:<10} {:<13} {}\n",
                truncate_pad(&name, 15),
                server.source.as_str(),
                transport_label(&server.config.transport),
                status,
                {
                    let visible = server
                        .tools
                        .iter()
                        .filter(|tool| self.policy.allows(server.source, &name, &tool.name))
                        .count();
                    let total = server.tools.len();
                    if total == 0 || visible == total {
                        total.to_string()
                    } else {
                        format!("{visible}/{total}")
                    }
                },
            ));
            if !extra.is_empty() {
                out.push_str(&format!("    {extra}\n"));
            }
        }
        out.push_str(
            "  /mcp <name> reconnect | enable | disable | allow <tool> | deny <tool>\n  /mcp add <name> --stdio <cmd> [args…] | --http <url>\n  /mcp pipe — provider mcp_url inspector\n",
        );
        out
    }

    /// Apply `/mcp` admin args: status, reconnect, enable, disable, allow, deny, add.
    pub async fn admin(&mut self, args: &str) -> Result<String, McpError> {
        match parse_mcp_admin(args) {
            McpAdminCmd::Status => Ok(self.status_table()),
            McpAdminCmd::PipeInspect => Ok(String::new()),
            McpAdminCmd::Reconnect(name) => {
                self.reconnect(&name).await?;
                Ok(format!("reconnected '{name}'\n{}", self.status_table()))
            }
            McpAdminCmd::Enable(name) => {
                self.set_enabled(&name, true)?;
                Ok(format!(
                    "enabled '{name}' (not connected until first use_tool)\n{}",
                    self.status_table()
                ))
            }
            McpAdminCmd::Disable(name) => {
                self.set_enabled(&name, false)?;
                self.disconnect_transport(&name).await;
                Ok(format!("disabled '{name}'\n{}", self.status_table()))
            }
            McpAdminCmd::Allow { server, tool } => self.admin_allow(&server, &tool),
            McpAdminCmd::Deny { server, tool } => self.admin_deny(&server, &tool),
            McpAdminCmd::AddStdio {
                name,
                command,
                args,
            } => {
                let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
                let config = McpServerConfig::stdio(name, command, &arg_refs);
                self.admin_add(config)
            }
            McpAdminCmd::AddHttp { name, url } => self.admin_add(McpServerConfig::http(name, url)),
            McpAdminCmd::Usage(message) => Ok(format!("{message}\n{}", self.status_table())),
        }
    }

    fn admin_allow(&mut self, server: &str, tool: &str) -> Result<String, McpError> {
        let source = self
            .servers
            .get(server)
            .ok_or_else(|| McpError::ServerNotFound(server.to_string()))?
            .source;
        self.policy
            .allow_tool(source, server, tool)
            .map_err(McpError::Server)?;
        let persist = self.persist_json_policy(server);
        Ok(format!(
            "allowed '{tool}' on '{server}'{persist}\n{}",
            self.status_table()
        ))
    }

    fn admin_deny(&mut self, server: &str, tool: &str) -> Result<String, McpError> {
        let source = self
            .servers
            .get(server)
            .ok_or_else(|| McpError::ServerNotFound(server.to_string()))?
            .source;
        self.policy
            .deny_tool(source, server, tool)
            .map_err(McpError::Server)?;
        let persist = self.persist_json_policy(server);
        Ok(format!(
            "denied '{tool}' on '{server}'{persist}\n{}",
            self.status_table()
        ))
    }

    fn admin_add(&mut self, config: McpServerConfig) -> Result<String, McpError> {
        let name = config.name.clone();
        if self.servers.contains_key(&name) {
            return Err(McpError::Server(format!(
                "server '{name}' is already registered"
            )));
        }
        let Some(root) = self.workspace_root.clone() else {
            return Err(McpError::Server(
                "no workspace root; cannot write .hi/mcp".into(),
            ));
        };
        write_hi_mcp_server(&root, &config)?;
        self.register(DiscoveredMcpServer {
            config,
            source: McpConfigSource::Hi,
            enabled: true,
            blocked_reason: None,
        });
        Ok(format!(
            "added '{name}' → .hi/mcp/{name}.json (connects on first use_tool)\n{}",
            self.status_table()
        ))
    }

    /// Persist only/exclude into `.hi/mcp/<name>.json` when that file exists.
    fn persist_json_policy(&self, server: &str) -> String {
        let Some(root) = &self.workspace_root else {
            return " (this session only)".into();
        };
        let path = hi_mcp_json_path(root, server);
        if !path.is_file() {
            return String::new();
        }
        let list = self.policy.server_list(server);
        match merge_allowlist_into_server_json(&path, &list.only, &list.exclude) {
            Ok(()) => format!(" (saved {})", path.display()),
            Err(err) => format!(" (could not save {}: {err})", path.display()),
        }
    }

    /// Time a connect + tools/list (for `hi mcp test`).
    pub async fn test_server(&mut self, name: &str) -> Result<String, McpError> {
        let started = std::time::Instant::now();
        self.ensure_connected(name, Duration::from_secs(8)).await?;
        let elapsed = started.elapsed();
        let tools = self.list_tools(name)?;
        Ok(format!(
            "mcp test '{name}': connected in {}ms, {} tool(s)\n",
            elapsed.as_millis(),
            tools.len()
        ))
    }

    /// Invoke a tool on a server.
    ///
    pub async fn invoke_tool(
        &mut self,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        // Admission precedes connection. Connecting may itself spawn a
        // repo-configured process or send expanded credential headers to an
        // HTTP endpoint, so a denied tool must not get that side effect merely
        // because it was invoked by name.
        let source = self
            .servers
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?
            .source;
        if let Some(reason) = self.policy.deny_reason(source, server_name, tool_name) {
            return Err(McpError::ToolInvocation(reason));
        }
        if let Err(err) = self
            .ensure_connected(server_name, Self::LAZY_CONNECT_GRACE)
            .await
        {
            let auto = self
                .servers
                .get(server_name)
                .is_some_and(|s| s.config.auto_reconnect && s.enabled);
            if auto {
                let _ = self
                    .ensure_connected(server_name, Self::LAZY_CONNECT_GRACE)
                    .await;
            }
            if self
                .servers
                .get(server_name)
                .is_none_or(|s| s.status != ServerStatus::Connected)
            {
                return Err(McpError::NotConnected(format!(
                    "server '{server_name}' is not connected ({err}). Try `/mcp {server_name} reconnect`"
                )));
            }
        }
        let server = self
            .servers
            .get_mut(server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;
        // Check the tool exists.
        if !server.tools.iter().any(|t| t.name == tool_name) {
            return Err(McpError::ToolInvocation(format!(
                "tool '{tool_name}' not found on server '{server_name}'"
            )));
        }
        let schema = server
            .tools
            .iter()
            .find(|tool| tool.name == tool_name)
            .map(|tool| tool.input_schema.clone())
            .ok_or_else(|| {
                McpError::ToolInvocation(format!(
                    "tool '{tool_name}' not found on server '{server_name}'"
                ))
            })?;
        validate_arguments(&schema, &arguments)?;
        let transport = server.transport.as_mut().ok_or_else(|| {
            McpError::NotConnected(format!(
                "server '{server_name}' is not connected. Try `/mcp {server_name} reconnect`"
            ))
        })?;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            transport.request(
                "tools/call",
                Some(serde_json::json!({"name": tool_name, "arguments": arguments})),
            ),
        )
        .await
        .map_err(|_| McpError::ToolInvocation("MCP tool call timed out".into()))??;
        let result = tool_result_from_value(&result);
        if result.content.len() > 2 * 1024 * 1024 {
            return Err(McpError::ToolInvocation(
                "MCP tool result exceeds 2 MiB output limit".into(),
            ));
        }
        Ok(result)
    }

    /// Read a resource from a server.
    pub async fn read_resource(
        &mut self,
        server_name: &str,
        uri: &str,
    ) -> Result<String, McpError> {
        let server = self
            .servers
            .get_mut(server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;
        if server.status != ServerStatus::Connected {
            return Err(McpError::NotConnected(server_name.to_string()));
        }
        // Check the resource exists.
        if !server.resources.iter().any(|r| r.uri == uri) {
            return Err(McpError::Server(format!(
                "resource '{uri}' not found on server '{server_name}'"
            )));
        }
        let transport = server
            .transport
            .as_mut()
            .ok_or_else(|| McpError::NotConnected(server_name.to_string()))?;
        let result = transport
            .request("resources/read", Some(serde_json::json!({"uri": uri})))
            .await?;
        Ok(clip_mcp_text(
            &resource_text_from_value(&result),
            MAX_MCP_RESOURCE_CHARS,
        ))
    }

    /// Check liveness of all connected servers.
    pub async fn check_liveness(&mut self) -> HashMap<String, ServerStatus> {
        self.servers
            .iter()
            .map(|(name, server)| (name.clone(), server.status))
            .collect()
    }
}

/// One newline-delimited JSON-RPC frame. A hostile or buggy MCP server can
/// emit a gigabyte line; `BufReader::lines()` would retain it all.
const MAX_RPC_LINE_BYTES: usize = 2 * 1024 * 1024;
const MAX_MCP_TOOLS: usize = 128;
const MAX_MCP_TOOL_DESCRIPTION_CHARS: usize = 2_000;
const MAX_MCP_RESOURCE_CHARS: usize = 64 * 1024;

async fn next_rpc_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
) -> Result<Option<String>, McpError> {
    let mut buf = Vec::new();
    loop {
        let data = reader.fill_buf().await?;
        if data.is_empty() {
            if buf.is_empty() {
                return Ok(None);
            }
            break;
        }
        if let Some(pos) = data.iter().position(|&b| b == b'\n') {
            if buf.len().saturating_add(pos) > MAX_RPC_LINE_BYTES {
                return Err(McpError::Transport(
                    "MCP JSON-RPC line exceeds 2 MiB".into(),
                ));
            }
            buf.extend_from_slice(&data[..pos]);
            reader.consume(pos + 1);
            break;
        }
        if buf.len().saturating_add(data.len()) > MAX_RPC_LINE_BYTES {
            return Err(McpError::Transport(
                "MCP JSON-RPC line exceeds 2 MiB".into(),
            ));
        }
        let n = data.len();
        buf.extend_from_slice(data);
        reader.consume(n);
    }
    if buf.last() == Some(&b'\r') {
        buf.pop();
    }
    Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
}

fn list_tools_from_result(value: serde_json::Value) -> Result<Vec<McpTool>, McpError> {
    let tools = value
        .get("tools")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Array(Vec::new()));
    let mut tools: Vec<McpTool> = serde_json::from_value(tools)?;
    if tools.len() > MAX_MCP_TOOLS {
        tools.truncate(MAX_MCP_TOOLS);
    }
    for tool in &mut tools {
        if tool.description.chars().count() > MAX_MCP_TOOL_DESCRIPTION_CHARS {
            tool.description = tool
                .description
                .chars()
                .take(MAX_MCP_TOOL_DESCRIPTION_CHARS.saturating_sub(1))
                .collect::<String>()
                + "…";
        }
    }
    Ok(tools)
}

fn validate_arguments(
    schema: &serde_json::Value,
    arguments: &serde_json::Value,
) -> Result<(), McpError> {
    let validator = jsonschema::draft202012::options()
        .build(schema)
        .map_err(|error| McpError::ToolInvocation(format!("invalid MCP input schema: {error}")))?;
    validator
        .validate(arguments)
        .map_err(|error| McpError::ToolInvocation(format!("invalid MCP tool arguments: {error}")))
}

fn list_resources_from_result(value: serde_json::Value) -> Result<Vec<McpResource>, McpError> {
    let resources = value
        .get("resources")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Array(Vec::new()));
    Ok(serde_json::from_value(resources)?)
}

fn tool_result_from_value(value: &serde_json::Value) -> McpToolResult {
    let content = value
        .get("content")
        .and_then(serde_json::Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(serde_json::Value::as_str) == Some("text") {
                        block.get("text").and_then(serde_json::Value::as_str)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| value.to_string());
    McpToolResult {
        content,
        is_error: value
            .get("isError")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    }
}

fn resource_text_from_value(value: &serde_json::Value) -> String {
    value
        .get("contents")
        .and_then(serde_json::Value::as_array)
        .map(|contents| {
            contents
                .iter()
                .filter_map(|content| content.get("text").and_then(serde_json::Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_else(|| value.to_string())
}

fn clip_mcp_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{clipped}…")
}

fn truncate_pad(s: &str, width: usize) -> String {
    let mut chars: String = s.chars().take(width).collect();
    while chars.chars().count() < width {
        chars.push(' ');
    }
    chars
}

impl Default for McpClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Discover MCP server configurations from a project's `.hi/mcp/` directory.
///
/// Each `.json` file in the directory describes one server. The file name
/// (without extension) becomes the server name.
pub fn discover_servers(project_dir: &Path) -> Vec<McpServerConfig> {
    discover_hi_servers(project_dir)
}

/// `.hi/mcp/*.json` only (highest-precedence source when merging).
pub fn discover_hi_servers(project_dir: &Path) -> Vec<McpServerConfig> {
    let mcp_dir = project_dir.join(".hi").join("mcp");
    let mut configs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&mcp_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                if let Ok(data) = std::fs::read_to_string(&path)
                    && let Ok(config) = parse_server_config(&name, &data)
                {
                    configs.push(config);
                }
            }
        }
    }
    configs
}

/// Parse a server config from JSON.
pub(crate) fn parse_server_config(
    name: &str,
    json: &str,
) -> Result<McpServerConfig, serde_json::Error> {
    #[derive(serde::Deserialize)]
    struct RawConfig {
        #[serde(default)]
        command: Option<String>,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        url: Option<String>,
        #[serde(default)]
        env: HashMap<String, String>,
        #[serde(default)]
        headers: HashMap<String, String>,
        #[serde(default)]
        auto_reconnect: Option<bool>,
        #[serde(default)]
        only: Vec<String>,
        #[serde(default)]
        exclude: Vec<String>,
    }

    let raw: RawConfig = serde_json::from_str(json)?;
    let transport = if let Some(command) = raw.command {
        McpTransport::Stdio {
            command,
            args: raw.args,
            env: raw.env,
        }
    } else if let Some(url) = raw.url {
        McpTransport::Http {
            url,
            headers: raw.headers,
        }
    } else {
        McpTransport::Stdio {
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
        }
    };

    Ok(McpServerConfig {
        name: name.to_string(),
        transport,
        auto_reconnect: raw.auto_reconnect.unwrap_or(true),
        only: raw.only,
        exclude: raw.exclude,
    })
}

const MCP_ADD_RESERVED: &[&str] = &[
    "add",
    "reconnect",
    "enable",
    "disable",
    "allow",
    "deny",
    "status",
    "list",
    "pipe",
    "test",
    "serve",
];

fn valid_mcp_server_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphanumeric()
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
        && !MCP_ADD_RESERVED.contains(&name)
}

/// Parsed `/mcp` admin arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpAdminCmd {
    Status,
    /// Exact `/mcp pipe` — frontend inspects the provider mcp_url.
    PipeInspect,
    Reconnect(String),
    Enable(String),
    Disable(String),
    Allow {
        server: String,
        tool: String,
    },
    Deny {
        server: String,
        tool: String,
    },
    AddStdio {
        name: String,
        command: String,
        args: Vec<String>,
    },
    AddHttp {
        name: String,
        url: String,
    },
    Usage(String),
}

/// Parse `/mcp` rest args.
pub fn parse_mcp_admin(args: &str) -> McpAdminCmd {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.is_empty() {
        return McpAdminCmd::Status;
    }
    if parts[0] == "add" {
        return parse_mcp_add(&parts[1..]);
    }
    if parts.len() == 1 && parts[0] == "pipe" {
        return McpAdminCmd::PipeInspect;
    }
    let (name, action, rest): (&str, &str, &[&str]) =
        if matches!(parts[0], "reconnect" | "enable" | "disable") {
            (
                parts.get(1).copied().unwrap_or(""),
                parts[0],
                parts.get(2..).unwrap_or(&[]),
            )
        } else {
            (
                parts[0],
                parts.get(1).copied().unwrap_or("status"),
                parts.get(2..).unwrap_or(&[]),
            )
        };
    if name.is_empty() || matches!(action, "status" | "list") {
        return McpAdminCmd::Status;
    }
    match action {
        "reconnect" => McpAdminCmd::Reconnect(name.into()),
        "enable" => McpAdminCmd::Enable(name.into()),
        "disable" => McpAdminCmd::Disable(name.into()),
        "allow" => {
            let tool = rest.first().copied().unwrap_or("");
            if tool.is_empty() {
                McpAdminCmd::Usage("usage: /mcp <name> allow <tool>".into())
            } else {
                McpAdminCmd::Allow {
                    server: name.into(),
                    tool: tool.into(),
                }
            }
        }
        "deny" => {
            let tool = rest.first().copied().unwrap_or("");
            if tool.is_empty() {
                McpAdminCmd::Usage("usage: /mcp <name> deny <tool>".into())
            } else {
                McpAdminCmd::Deny {
                    server: name.into(),
                    tool: tool.into(),
                }
            }
        }
        other => McpAdminCmd::Usage(format!(
            "unknown mcp action '{other}'\nusage: /mcp [pipe|<name> reconnect|enable|disable|allow <tool>|deny <tool>]\n       /mcp add <name> --stdio <cmd> [args…] | --http <url>"
        )),
    }
}

fn parse_mcp_add(parts: &[&str]) -> McpAdminCmd {
    let usage = "usage: /mcp add <name> --stdio <command> [args…] | --http <url>";
    let Some(name) = parts.first().copied() else {
        return McpAdminCmd::Usage(usage.into());
    };
    if !valid_mcp_server_name(name) {
        return McpAdminCmd::Usage(format!(
            "invalid MCP server name {name:?}; use letters, digits, '.', '_', '-'"
        ));
    }
    let Some(kind) = parts.get(1).copied() else {
        return McpAdminCmd::Usage(usage.into());
    };
    match kind {
        "--stdio" | "stdio" => {
            let Some(command) = parts.get(2).copied().filter(|c| !c.is_empty()) else {
                return McpAdminCmd::Usage(usage.into());
            };
            McpAdminCmd::AddStdio {
                name: name.into(),
                command: command.into(),
                args: parts[3..].iter().map(|s| (*s).to_string()).collect(),
            }
        }
        "--http" | "http" => {
            let Some(url) = parts.get(2).copied().filter(|u| !u.is_empty()) else {
                return McpAdminCmd::Usage(usage.into());
            };
            McpAdminCmd::AddHttp {
                name: name.into(),
                url: url.into(),
            }
        }
        _ => McpAdminCmd::Usage(usage.into()),
    }
}

pub fn hi_mcp_json_path(workspace_root: &Path, name: &str) -> PathBuf {
    workspace_root
        .join(".hi")
        .join("mcp")
        .join(format!("{name}.json"))
}

/// Write a new `.hi/mcp/<name>.json`. Fails if the file already exists.
pub fn write_hi_mcp_server(
    workspace_root: &Path,
    config: &McpServerConfig,
) -> Result<PathBuf, McpError> {
    if !valid_mcp_server_name(&config.name) {
        return Err(McpError::Server(format!(
            "invalid MCP server name {:?}",
            config.name
        )));
    }
    let dir = workspace_root.join(".hi").join("mcp");
    std::fs::create_dir_all(&dir)?;
    let path = hi_mcp_json_path(workspace_root, &config.name);
    if path.exists() {
        return Err(McpError::Server(format!(
            "already exists: {}",
            path.display()
        )));
    }
    let mut obj = serde_json::Map::new();
    match &config.transport {
        McpTransport::Stdio { command, args, env } => {
            obj.insert("command".into(), serde_json::json!(command));
            if !args.is_empty() {
                obj.insert("args".into(), serde_json::json!(args));
            }
            if !env.is_empty() {
                obj.insert("env".into(), serde_json::json!(env));
            }
        }
        McpTransport::Http { url, headers } => {
            obj.insert("url".into(), serde_json::json!(url));
            if !headers.is_empty() {
                obj.insert("headers".into(), serde_json::json!(headers));
            }
        }
    }
    if !config.only.is_empty() {
        obj.insert("only".into(), serde_json::json!(config.only));
    }
    if !config.exclude.is_empty() {
        obj.insert("exclude".into(), serde_json::json!(config.exclude));
    }
    if !config.auto_reconnect {
        obj.insert("auto_reconnect".into(), serde_json::json!(false));
    }
    let encoded = serde_json::to_string_pretty(&serde_json::Value::Object(obj))?;
    std::fs::write(&path, encoded)?;
    Ok(path)
}

/// Merge `only` / `exclude` into an existing server JSON without dropping other keys.
pub fn merge_allowlist_into_server_json(
    path: &Path,
    only: &[String],
    exclude: &[String],
) -> Result<(), McpError> {
    let text = std::fs::read_to_string(path)?;
    let mut value: serde_json::Value = serde_json::from_str(&text)?;
    let obj = value
        .as_object_mut()
        .ok_or_else(|| McpError::Server(format!("{} is not a JSON object", path.display())))?;
    if only.is_empty() {
        obj.remove("only");
    } else {
        obj.insert("only".into(), serde_json::json!(only));
    }
    if exclude.is_empty() {
        obj.remove("exclude");
    } else {
        obj.insert("exclude".into(), serde_json::json!(exclude));
    }
    std::fs::write(path, serde_json::to_string_pretty(&value)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_server_config() -> McpServerConfig {
        // A tiny newline-framed JSON-RPC server keeps transport tests
        // independent of Node/npm and exercises the real handshake.
        let script = concat!(
            "while IFS= read -r line; do\n",
            "case \"$line\" in\n",
            "*'\"method\":\"initialize\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"serverInfo\":{\"name\":\"fake\",\"version\":\"1\"},\"capabilities\":{\"tools\":{},\"resources\":{}}}}\\n' ;;\n",
            "*'\"method\":\"tools/list\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"echo input\",\"inputSchema\":{\"type\":\"object\"}}]}}\\n' ;;\n",
            "*'\"method\":\"resources/list\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"resources\":[]}}\\n' ;;\n",
            "*'\"method\":\"tools/call\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":4,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}}\\n' ;;\n",
            "esac\ndone"
        );
        McpServerConfig::stdio("test", "sh", &["-c", script])
    }

    #[test]
    fn server_config_stdio() {
        let config = McpServerConfig::stdio("test", "npx", &["-y", "@mcp/server"]);
        assert_eq!(config.name, "test");
        match config.transport {
            McpTransport::Stdio { command, args, .. } => {
                assert_eq!(command, "npx");
                assert_eq!(args, vec!["-y", "@mcp/server"]);
            }
            _ => panic!("expected Stdio transport"),
        }
    }

    #[test]
    fn server_config_http() {
        let config = McpServerConfig::http("test", "https://example.com/mcp");
        assert_eq!(config.name, "test");
        match config.transport {
            McpTransport::Http { url, .. } => {
                assert_eq!(url, "https://example.com/mcp");
            }
            _ => panic!("expected Http transport"),
        }
    }

    #[test]
    fn server_config_with_env() {
        let config = McpServerConfig::stdio("test", "cmd", &[]).with_env("API_KEY", "secret");
        if let McpTransport::Stdio { env, .. } = config.transport {
            assert_eq!(env.get("API_KEY"), Some(&"secret".to_string()));
        }
    }

    #[test]
    fn server_config_with_header() {
        let config = McpServerConfig::http("test", "https://example.com")
            .with_header("Authorization", "Bearer token");
        if let McpTransport::Http { headers, .. } = config.transport {
            assert_eq!(
                headers.get("Authorization"),
                Some(&"Bearer token".to_string())
            );
        }
    }

    #[test]
    fn mcp_tool_serde_roundtrip() {
        let tool = McpTool {
            name: "search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let json = serde_json::to_string(&tool).unwrap();
        let back: McpTool = serde_json::from_str(&json).unwrap();
        assert_eq!(tool, back);
    }

    #[test]
    fn mcp_resource_serde_roundtrip() {
        let resource = McpResource {
            uri: "file:///test".into(),
            name: "test".into(),
            description: Some("a test".into()),
            mime_type: Some("text/plain".into()),
        };
        let json = serde_json::to_string(&resource).unwrap();
        let back: McpResource = serde_json::from_str(&json).unwrap();
        assert_eq!(resource, back);
    }

    #[test]
    fn mcp_resource_optional_fields() {
        let resource = McpResource {
            uri: "file:///test".into(),
            name: "test".into(),
            description: None,
            mime_type: None,
        };
        let json = serde_json::to_string(&resource).unwrap();
        assert!(!json.contains("description"));
        assert!(!json.contains("mime_type"));
    }

    #[tokio::test]
    async fn client_connect_and_disconnect() {
        let mut client = McpClient::new();
        let config = fake_server_config();
        client.connect(config).await.unwrap();
        assert!(client.server_names().contains(&"test".to_string()));
        assert_eq!(client.status("test").unwrap(), ServerStatus::Connected);
        assert_eq!(client.list_tools("test").unwrap()[0].name, "echo");
        client.disconnect("test").await.unwrap();
        assert!(!client.server_names().contains(&"test".to_string()));
    }

    #[tokio::test]
    async fn client_disconnect_nonexistent_fails() {
        let mut client = McpClient::new();
        let result = client.disconnect("nonexistent").await;
        assert!(matches!(result, Err(McpError::ServerNotFound(_))));
    }

    #[tokio::test]
    async fn client_status_nonexistent_fails() {
        let client = McpClient::new();
        let result = client.status("nonexistent");
        assert!(matches!(result, Err(McpError::ServerNotFound(_))));
    }

    #[tokio::test]
    async fn client_list_tools_nonexistent_fails() {
        let client = McpClient::new();
        let result = client.list_tools("nonexistent");
        assert!(matches!(result, Err(McpError::ServerNotFound(_))));
    }

    #[tokio::test]
    async fn client_invoke_tool() {
        let mut client = McpClient::new();
        let config = fake_server_config();
        client.connect(config).await.unwrap();
        let result = client
            .invoke_tool("test", "echo", serde_json::json!({"value":"x"}))
            .await;
        let result = result.unwrap();
        assert_eq!(result.content, "ok");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn client_check_liveness() {
        let mut client = McpClient::new();
        let config = fake_server_config();
        client.connect(config).await.unwrap();
        let statuses = client.check_liveness().await;
        assert!(statuses.contains_key("test"));
    }

    #[test]
    fn discover_servers_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let configs = discover_servers(tmp.path());
        assert!(configs.is_empty());
    }

    #[test]
    fn discover_servers_finds_json() {
        let tmp = tempfile::tempdir().unwrap();
        let mcp_dir = tmp.path().join(".hi").join("mcp");
        std::fs::create_dir_all(&mcp_dir).unwrap();
        std::fs::write(
            mcp_dir.join("my-server.json"),
            r#"{"command":"npx","args":["-y","@mcp/server"]}"#,
        )
        .unwrap();
        let configs = discover_servers(tmp.path());
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "my-server");
    }

    #[test]
    fn discover_servers_ignores_non_json() {
        let tmp = tempfile::tempdir().unwrap();
        let mcp_dir = tmp.path().join(".hi").join("mcp");
        std::fs::create_dir_all(&mcp_dir).unwrap();
        std::fs::write(mcp_dir.join("readme.txt"), "hello").unwrap();
        std::fs::write(mcp_dir.join("bad.json"), "not valid json").unwrap();
        let configs = discover_servers(tmp.path());
        assert!(configs.is_empty());
    }

    #[test]
    fn list_tools_caps_count_and_description() {
        let tools: Vec<serde_json::Value> = (0..200)
            .map(|i| {
                serde_json::json!({
                    "name": format!("t{i}"),
                    "description": "D".repeat(5_000),
                    "inputSchema": {"type": "object"}
                })
            })
            .collect();
        let listed = list_tools_from_result(serde_json::json!({"tools": tools})).unwrap();
        assert_eq!(listed.len(), MAX_MCP_TOOLS);
        assert!(
            listed
                .iter()
                .all(|tool| tool.description.chars().count() <= MAX_MCP_TOOL_DESCRIPTION_CHARS)
        );
    }

    #[test]
    fn resource_text_is_clipped() {
        let text = clip_mcp_text(
            &"R".repeat(MAX_MCP_RESOURCE_CHARS + 80),
            MAX_MCP_RESOURCE_CHARS,
        );
        assert!(text.chars().count() <= MAX_MCP_RESOURCE_CHARS);
        assert!(text.ends_with('…'));
    }

    #[tokio::test]
    async fn next_rpc_line_rejects_oversize_and_reads_normal_frames() {
        let data = b"{\"ok\":true}\n{\"next\":1}\n";
        let mut reader = BufReader::new(&data[..]);
        assert_eq!(
            next_rpc_line(&mut reader).await.unwrap().as_deref(),
            Some("{\"ok\":true}")
        );
        assert_eq!(
            next_rpc_line(&mut reader).await.unwrap().as_deref(),
            Some("{\"next\":1}")
        );
        assert!(next_rpc_line(&mut reader).await.unwrap().is_none());

        let huge = vec![b'x'; MAX_RPC_LINE_BYTES + 8];
        let mut reader = BufReader::new(&huge[..]);
        let err = next_rpc_line(&mut reader).await.unwrap_err();
        assert!(err.to_string().contains("2 MiB"), "{err}");
    }

    #[test]
    fn parse_server_config_stdio() {
        let json = r#"{"command":"npx","args":["-y","@mcp/server"],"env":{"KEY":"val"}}"#;
        let config = parse_server_config("test", json).unwrap();
        assert_eq!(config.name, "test");
        match config.transport {
            McpTransport::Stdio { command, args, env } => {
                assert_eq!(command, "npx");
                assert_eq!(args, vec!["-y", "@mcp/server"]);
                assert_eq!(env.get("KEY"), Some(&"val".to_string()));
            }
            _ => panic!("expected Stdio"),
        }
    }

    #[test]
    fn parse_server_config_http() {
        let json = r#"{"url":"https://example.com/mcp","headers":{"Auth":"Bearer x"}}"#;
        let config = parse_server_config("test", json).unwrap();
        match config.transport {
            McpTransport::Http { url, headers } => {
                assert_eq!(url, "https://example.com/mcp");
                assert_eq!(headers.get("Auth"), Some(&"Bearer x".to_string()));
            }
            _ => panic!("expected Http"),
        }
    }

    #[test]
    fn parse_server_config_auto_reconnect_default() {
        let json = r#"{"command":"echo","args":[]}"#;
        let config = parse_server_config("test", json).unwrap();
        assert!(config.auto_reconnect);
    }

    #[test]
    fn parse_server_config_auto_reconnect_false() {
        let json = r#"{"command":"echo","args":[],"auto_reconnect":false}"#;
        let config = parse_server_config("test", json).unwrap();
        assert!(!config.auto_reconnect);
    }

    #[test]
    fn parse_server_config_only_exclude() {
        let json = r#"{"command":"npx","args":["-y","x"],"only":["search"],"exclude":["delete"]}"#;
        let config = parse_server_config("docs", json).unwrap();
        assert_eq!(config.only, vec!["search"]);
        assert_eq!(config.exclude, vec!["delete"]);
    }

    #[test]
    fn parse_mcp_admin_add_allow_deny() {
        assert_eq!(parse_mcp_admin(""), McpAdminCmd::Status);
        assert_eq!(parse_mcp_admin("pipe"), McpAdminCmd::PipeInspect);
        assert_eq!(
            parse_mcp_admin("docs reconnect"),
            McpAdminCmd::Reconnect("docs".into())
        );
        assert_eq!(
            parse_mcp_admin("docs allow search"),
            McpAdminCmd::Allow {
                server: "docs".into(),
                tool: "search".into(),
            }
        );
        assert_eq!(
            parse_mcp_admin("pipe deny pipe.usage.summary"),
            McpAdminCmd::Deny {
                server: "pipe".into(),
                tool: "pipe.usage.summary".into(),
            }
        );
        assert_eq!(
            parse_mcp_admin("add docs --stdio npx -y @mcp/server"),
            McpAdminCmd::AddStdio {
                name: "docs".into(),
                command: "npx".into(),
                args: vec!["-y".into(), "@mcp/server".into()],
            }
        );
        assert_eq!(
            parse_mcp_admin("add remote --http https://example.com/mcp"),
            McpAdminCmd::AddHttp {
                name: "remote".into(),
                url: "https://example.com/mcp".into(),
            }
        );
        assert!(matches!(
            parse_mcp_admin("add add --stdio x"),
            McpAdminCmd::Usage(_)
        ));
    }

    #[tokio::test]
    async fn admin_add_writes_json_and_registers() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = hi_tools::ProcessRunner::new(tmp.path()).unwrap();
        let mut client = McpClient::with_process_runner(runner);
        let out = client
            .admin("add docs --http https://example.test/mcp")
            .await
            .unwrap();
        assert!(out.contains("added 'docs'"), "{out}");
        let path = hi_mcp_json_path(tmp.path(), "docs");
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("https://example.test/mcp"), "{body}");
        assert!(client.server_names().contains(&"docs".to_string()));
    }

    #[tokio::test]
    async fn admin_deny_persists_exclude_in_json() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = hi_tools::ProcessRunner::new(tmp.path()).unwrap();
        let mut client = McpClient::with_process_runner(runner);
        client
            .admin("add docs --http https://example.test/mcp")
            .await
            .unwrap();
        let out = client.admin("docs deny wipe").await.unwrap();
        assert!(out.contains("denied 'wipe'"), "{out}");
        let body = std::fs::read_to_string(hi_mcp_json_path(tmp.path(), "docs")).unwrap();
        assert!(body.contains("wipe"), "{body}");
        assert!(!client.policy().allows(McpConfigSource::Hi, "docs", "wipe"));
    }

    #[test]
    fn server_status_equality() {
        assert_eq!(ServerStatus::Connected, ServerStatus::Connected);
        assert_ne!(ServerStatus::Connected, ServerStatus::Disconnected);
    }

    #[test]
    fn mcp_client_default() {
        let client = McpClient::default();
        assert!(client.server_names().is_empty());
    }

    #[test]
    fn register_applies_json_exclude() {
        let mut client = McpClient::new();
        let mut config = McpServerConfig::http("docs", "https://example.test/mcp");
        config.exclude = vec!["wipe".into()];
        client.register(DiscoveredMcpServer {
            config,
            source: McpConfigSource::Hi,
            enabled: true,
            blocked_reason: None,
        });
        assert!(!client.policy().allows(McpConfigSource::Hi, "docs", "wipe"));
        assert!(
            client
                .policy()
                .allows(McpConfigSource::Hi, "docs", "search")
        );
    }

    #[test]
    fn blocked_server_cannot_be_silently_enabled() {
        let mut client = McpClient::new();
        client.register(DiscoveredMcpServer {
            config: McpServerConfig::stdio("blocked", "true", &[]),
            source: McpConfigSource::Claude,
            enabled: false,
            blocked_reason: Some("excluded".into()),
        });
        let err = client.set_enabled("blocked", true).unwrap_err();
        assert!(err.to_string().contains("excluded"), "{err}");
        assert_eq!(client.status("blocked").unwrap(), ServerStatus::Disabled);
    }

    #[tokio::test]
    async fn denied_tool_does_not_connect_or_spawn_server() {
        let mut client = McpClient::new();
        let mut config = McpServerConfig::stdio("denied", "sleep", &["30"]);
        config.exclude = vec!["echo".into()];
        client.register(DiscoveredMcpServer {
            config,
            source: McpConfigSource::Hi,
            enabled: true,
            blocked_reason: None,
        });

        let error = client
            .invoke_tool("denied", "echo", serde_json::json!({}))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("denied"), "{error}");
        assert_eq!(client.status("denied").unwrap(), ServerStatus::Disconnected);
    }

    #[tokio::test]
    async fn lazy_connect_fail_fast_on_dead_server() {
        let mut client = McpClient::new();
        client.register(DiscoveredMcpServer {
            config: McpServerConfig::stdio("slow", "sleep", &["30"]),
            source: McpConfigSource::Hi,
            enabled: true,
            blocked_reason: None,
        });
        let started = std::time::Instant::now();
        let err = client
            .invoke_tool("slow", "echo", serde_json::json!({}))
            .await
            .unwrap_err();
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(6),
            "lazy connect waited too long: {elapsed:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("reconnect") || msg.contains("not connected"),
            "{msg}"
        );
    }

    #[tokio::test]
    async fn generation_guard_discards_stale_handshake() {
        let mut client = McpClient::new();
        client.register(DiscoveredMcpServer {
            config: fake_server_config(),
            source: McpConfigSource::Hi,
            enabled: true,
            blocked_reason: None,
        });
        client.connect_inner("test").await.unwrap();
        let generation = client.servers.get("test").unwrap().generation;
        assert!(generation >= 1);
        client.disconnect_transport("test").await;
        let generation2 = client.servers.get("test").unwrap().generation;
        assert!(generation2 > generation);
    }
}
