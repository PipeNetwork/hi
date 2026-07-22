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
use std::path::PathBuf;
use std::process::Command;

use async_trait::async_trait;
use thiserror::Error;

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
    #[serde(default)]
    pub input_schema: serde_json::Value,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
#[derive(Debug)]
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
}

impl McpClient {
    /// Create a new MCP client with no connected servers.
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(),
        }
    }

    /// Connect to an MCP server.
    ///
    /// In this stub implementation, this records the server config but does
    /// not actually connect. A full implementation would spawn the process
    /// (stdio) or open a WebSocket (HTTP), perform the MCP handshake, and
    /// discover tools.
    pub async fn connect(&mut self, config: McpServerConfig) -> Result<(), McpError> {
        let name = config.name.clone();
        let server = McpServer {
            config,
            status: ServerStatus::Connecting,
            server_name: None,
            server_version: None,
            tools: Vec::new(),
            resources: Vec::new(),
        };
        self.servers.insert(name, server);
        Ok(())
    }

    /// Disconnect from a server.
    pub async fn disconnect(&mut self, name: &str) -> Result<(), McpError> {
        self.servers
            .remove(name)
            .ok_or_else(|| McpError::ServerNotFound(name.to_string()))?;
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

    /// List resources from a server.
    pub fn list_resources(&self, name: &str) -> Result<&[McpResource], McpError> {
        self.servers
            .get(name)
            .map(|s| s.resources.as_slice())
            .ok_or_else(|| McpError::ServerNotFound(name.to_string()))
    }

    /// Invoke a tool on a server.
    ///
    /// In this stub, the tool invocation is not actually sent. A full
    /// implementation would send a `tools/call` JSON-RPC request.
    pub async fn invoke_tool(
        &mut self,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        let server = self
            .servers
            .get(server_name)
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
        // Stub: return the arguments as the content.
        Ok(McpToolResult {
            content: format!("{{\"tool\":\"{tool_name}\",\"arguments\":{arguments}}}"),
            is_error: false,
        })
    }

    /// Read a resource from a server.
    pub async fn read_resource(
        &mut self,
        server_name: &str,
        uri: &str,
    ) -> Result<String, McpError> {
        let server = self
            .servers
            .get(server_name)
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
        // Stub: return empty content.
        Ok(String::new())
    }

    /// Check liveness of all connected servers.
    pub async fn check_liveness(&mut self) -> HashMap<String, ServerStatus> {
        self.servers
            .iter()
            .map(|(name, server)| (name.clone(), server.status))
            .collect()
    }
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
pub fn discover_servers(project_dir: &PathBuf) -> Vec<McpServerConfig> {
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
                if let Ok(data) = std::fs::read_to_string(&path) {
                    if let Ok(config) = parse_server_config(&name, &data) {
                        configs.push(config);
                    }
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
        let config = McpServerConfig::stdio("test", "echo", &["hello"]);
        client.connect(config).await.unwrap();
        assert!(client.server_names().contains(&"test".to_string()));
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
    async fn client_invoke_tool_not_connected() {
        let mut client = McpClient::new();
        let config = McpServerConfig::stdio("test", "echo", &[]);
        client.connect(config).await.unwrap();
        // Server is in Connecting state, not Connected.
        let result = client
            .invoke_tool("test", "some_tool", serde_json::json!({}))
            .await;
        assert!(matches!(result, Err(McpError::NotConnected(_))));
    }

    #[tokio::test]
    async fn client_check_liveness() {
        let mut client = McpClient::new();
        let config = McpServerConfig::stdio("test", "echo", &[]);
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
