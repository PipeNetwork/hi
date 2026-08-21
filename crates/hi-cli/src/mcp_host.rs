//! Connect workspace MCP servers and expose them through search/select tools.
//!
//! Each MCP tool's JSON Schema stays off the model request. The agent sees
//! two gateway tools (`search_tool`, `use_tool`); `search_tool` returns names
//! and schemas on demand.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use hi_tools::{McpBackend, McpToolInfo};
use serde_json::Value;

pub struct ConnectedMcp {
    client: tokio::sync::Mutex<hi_mcp::McpClient>,
}

#[async_trait]
impl McpBackend for ConnectedMcp {
    async fn search(&self, query: Option<&str>) -> Result<Vec<McpToolInfo>> {
        let client = self.client.lock().await;
        let query = query.map(|q| q.to_ascii_lowercase());
        let mut out = Vec::new();
        for server in client.server_names() {
            let Ok(tools) = client.list_tools(&server) else {
                continue;
            };
            for tool in tools {
                if let Some(q) = query.as_deref() {
                    let name = tool.name.to_ascii_lowercase();
                    let description = tool.description.to_ascii_lowercase();
                    if !name.contains(q) && !description.contains(q) {
                        continue;
                    }
                }
                out.push(McpToolInfo {
                    server: server.clone(),
                    tool: tool.name.clone(),
                    description: tool.description.clone(),
                    schema: tool.input_schema.clone(),
                });
            }
        }
        Ok(out)
    }

    async fn call(&self, server: &str, tool: &str, arguments: &Value) -> Result<String> {
        let mut client = self.client.lock().await;
        let result = client
            .invoke_tool(server, tool, arguments.clone())
            .await
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        if result.is_error {
            anyhow::bail!("{}", result.content);
        }
        Ok(result.content)
    }
}

/// Discover `.hi/mcp/*.json`, connect stdio servers, and return a backend
/// when at least one handshake succeeds. Fail-open: a dead server is skipped.
pub async fn connect_workspace_mcp(workspace_root: &Path) -> Option<Arc<ConnectedMcp>> {
    if !matches!(
        hi_tools::folder_trust::resolve_trust(workspace_root),
        hi_tools::folder_trust::TrustOutcome::Trusted
    ) {
        return None;
    }
    let configs = hi_mcp::discover_servers(workspace_root);
    if configs.is_empty() {
        return None;
    }
    let runner = hi_tools::ProcessRunner::new(workspace_root).ok()?;
    let mut client = hi_mcp::McpClient::with_process_runner(runner);
    let mut connected = 0usize;
    for config in configs {
        let name = config.name.clone();
        if matches!(config.transport, hi_mcp::McpTransport::Http { .. }) {
            eprintln!("mcp: skipping {name}: HTTP transport is not implemented");
            continue;
        }
        match tokio::time::timeout(Duration::from_secs(8), client.connect(config)).await {
            Ok(Ok(())) => connected += 1,
            Ok(Err(err)) => eprintln!("mcp: failed to connect {name}: {err}"),
            Err(_) => eprintln!("mcp: timed out connecting {name}"),
        }
    }
    if connected == 0 {
        return None;
    }
    Some(Arc::new(ConnectedMcp {
        client: tokio::sync::Mutex::new(client),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fake_stdio_config() -> hi_mcp::McpServerConfig {
        let script = concat!(
            "while IFS= read -r line; do\n",
            "case \"$line\" in\n",
            "*'\"method\":\"initialize\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"serverInfo\":{\"name\":\"fake\",\"version\":\"1\"},\"capabilities\":{\"tools\":{},\"resources\":{}}}}\\n' ;;\n",
            "*'\"method\":\"tools/list\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"echo input\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}}}}]}}\\n' ;;\n",
            "*'\"method\":\"resources/list\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"resources\":[]}}\\n' ;;\n",
            "*'\"method\":\"tools/call\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":4,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}}\\n' ;;\n",
            "esac\ndone"
        );
        hi_mcp::McpServerConfig::stdio("demo", "sh", &["-c", script])
    }

    #[tokio::test]
    async fn search_and_call_go_through_the_gateway() {
        let runner = hi_tools::ProcessRunner::new(std::env::temp_dir()).unwrap();
        let mut client = hi_mcp::McpClient::with_process_runner(runner);
        client.connect(fake_stdio_config()).await.unwrap();
        let backend = ConnectedMcp {
            client: tokio::sync::Mutex::new(client),
        };
        let found = backend.search(Some("echo")).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].server, "demo");
        assert_eq!(found[0].tool, "echo");
        assert_eq!(found[0].schema["properties"]["text"]["type"], "string");
        let out = backend
            .call("demo", "echo", &json!({"text": "hi"}))
            .await
            .unwrap();
        assert_eq!(out, "ok");
    }

    #[tokio::test]
    async fn missing_mcp_dir_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(connect_workspace_mcp(tmp.path()).await.is_none());
    }
}
