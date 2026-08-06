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

use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;

use async_trait::async_trait;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
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
}

/// A connected MCP server.
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
    transport: Option<Box<dyn McpTransportTrait>>,
}

/// A line-delimited JSON-RPC transport for MCP servers launched as child
/// processes. MCP stdio servers must keep stdout protocol-clean; stderr is
/// intentionally discarded here because it is diagnostic, not protocol data.
struct StdioTransport {
    child: Child,
    stdin: ChildStdin,
    stdout: tokio::io::Lines<BufReader<ChildStdout>>,
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
            stdout: BufReader::new(stdout).lines(),
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
        while let Some(line) = self.stdout.next_line().await? {
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

/// The MCP client that manages server connections and tool calls.
pub struct McpClient {
    servers: HashMap<String, McpServer>,
    process_runner: Option<std::sync::Arc<hi_tools::ProcessRunner>>,
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
        }
    }

    /// Use hi's sanitized process/sandbox boundary for future stdio servers.
    pub fn with_process_runner(runner: hi_tools::ProcessRunner) -> Self {
        Self {
            servers: HashMap::new(),
            process_runner: Some(std::sync::Arc::new(runner)),
        }
    }

    /// Connect to an MCP server.
    ///
    pub async fn connect(&mut self, config: McpServerConfig) -> Result<(), McpError> {
        let name = config.name.clone();
        let mut transport: Box<dyn McpTransportTrait> = match &config.transport {
            McpTransport::Stdio { .. } => {
                Box::new(StdioTransport::spawn(&config, self.process_runner.as_deref()).await?)
            }
            McpTransport::Http { .. } => {
                return Err(McpError::Transport(
                    "HTTP MCP transport is not implemented yet; use stdio".into(),
                ));
            }
        };
        let initialize = transport
            .request(
                "initialize",
                Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "hi", "version": env!("CARGO_PKG_VERSION")}
                })),
            )
            .await?;
        transport.notify("notifications/initialized", None).await?;
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
        let server = McpServer {
            config,
            status: ServerStatus::Connected,
            server_name: initialize
                .get("serverInfo")
                .and_then(|value| value.get("name"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            server_version: initialize
                .get("serverInfo")
                .and_then(|value| value.get("version"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            tools,
            resources,
            transport: Some(transport),
        };
        self.servers.insert(name, server);
        Ok(())
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

    /// List tools from a server.
    pub fn list_tools(&self, name: &str) -> Result<&[McpTool], McpError> {
        self.servers
            .get(name)
            .map(|s| s.tools.as_slice())
            .ok_or_else(|| McpError::ServerNotFound(name.to_string()))
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

    /// Invoke a tool on a server.
    ///
    pub async fn invoke_tool(
        &mut self,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        let server = self
            .servers
            .get_mut(server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;
        if server.status != ServerStatus::Connected {
            return Err(McpError::NotConnected(server_name.to_string()));
        }
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
        let transport = server
            .transport
            .as_mut()
            .ok_or_else(|| McpError::NotConnected(server_name.to_string()))?;
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
        Ok(resource_text_from_value(&result))
    }

    /// Check liveness of all connected servers.
    pub async fn check_liveness(&mut self) -> HashMap<String, ServerStatus> {
        self.servers
            .iter()
            .map(|(name, server)| (name.clone(), server.status))
            .collect()
    }
}

fn list_tools_from_result(value: serde_json::Value) -> Result<Vec<McpTool>, McpError> {
    let tools = value
        .get("tools")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Array(Vec::new()));
    Ok(serde_json::from_value(tools)?)
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
                    && let Ok(config) = parse_server_config(&name, &data) {
                        configs.push(config);
                    }
            }
        }
    }
    configs
}

/// Parse a server config from JSON.
fn parse_server_config(name: &str, json: &str) -> Result<McpServerConfig, serde_json::Error> {
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
    })
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
        let configs = discover_servers(&tmp.path().to_path_buf());
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
        let configs = discover_servers(&tmp.path().to_path_buf());
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
        let configs = discover_servers(&tmp.path().to_path_buf());
        assert!(configs.is_empty());
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
    fn server_status_equality() {
        assert_eq!(ServerStatus::Connected, ServerStatus::Connected);
        assert_ne!(ServerStatus::Connected, ServerStatus::Disconnected);
    }

    #[test]
    fn mcp_client_default() {
        let client = McpClient::default();
        assert!(client.server_names().is_empty());
    }
}
