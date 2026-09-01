//! `hi mcp serve` — expose read/bash/edit/write over MCP stdio for other harnesses.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Context, Result};
use async_trait::async_trait;
use hi_mcp::{McpStdioHandler, McpTool, hi_serve_tools, serve_stdio};
use hi_tools::{BackgroundRegistry, ReadCache, RepoMapCache, ToolStatus, execute_in_runtime};
use serde_json::Value;

struct HiToolHandler {
    root: PathBuf,
    state: PathBuf,
    lsp: Arc<hi_lsp::LspManager>,
    background: BackgroundRegistry,
    read_cache: Mutex<ReadCache>,
    repo_map: Mutex<RepoMapCache>,
}

#[async_trait]
impl McpStdioHandler for HiToolHandler {
    fn tools(&self) -> Vec<McpTool> {
        hi_serve_tools()
    }

    async fn call(&mut self, name: &str, arguments: Value) -> Result<String, String> {
        if !matches!(name, "read" | "bash" | "edit" | "write") {
            return Err(format!("tool '{name}' is not exported by hi mcp serve"));
        }
        let args = arguments.to_string();
        let outcome = execute_in_runtime(
            &self.root,
            &self.state,
            &self.lsp,
            &self.background,
            &self.read_cache,
            &self.repo_map,
            name,
            &args,
        )
        .await;
        if outcome.status != ToolStatus::Succeeded {
            return Err(outcome.content);
        }
        Ok(outcome.content)
    }
}

pub(crate) async fn run() -> Result<()> {
    if !matches!(
        hi_tools::folder_trust::resolve_trust(
            &std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        ),
        hi_tools::folder_trust::TrustOutcome::Trusted
    ) {
        anyhow::bail!("folder is not trusted; refuse to serve workspace tools");
    }
    let root = std::env::current_dir().context("current directory")?;
    let root = root.canonicalize().unwrap_or(root);
    let state = root.join(".hi").join("mcp-serve");
    std::fs::create_dir_all(&state).ok();
    let lsp = Arc::new(hi_lsp::LspManager::new(&root).context("lsp manager")?);
    let handler = HiToolHandler {
        root,
        state,
        lsp,
        background: BackgroundRegistry::default(),
        read_cache: Mutex::new(ReadCache::new()),
        repo_map: Mutex::new(RepoMapCache::new()),
    };
    serve_stdio(handler).await.context("mcp stdio server")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_mcp::dispatch_line;

    #[tokio::test]
    async fn serve_handler_round_trip_read() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("note.txt"), "hello from hi").unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let state = root.join("state");
        std::fs::create_dir_all(&state).unwrap();
        let mut handler = HiToolHandler {
            lsp: Arc::new(hi_lsp::LspManager::new(&root).unwrap()),
            root,
            state,
            background: BackgroundRegistry::default(),
            read_cache: Mutex::new(ReadCache::new()),
            repo_map: Mutex::new(RepoMapCache::new()),
        };
        let listed = dispatch_line(
            &mut handler,
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
        )
        .await
        .unwrap();
        assert_eq!(listed["result"]["tools"].as_array().unwrap().len(), 4);
        let call = dispatch_line(
            &mut handler,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"read","arguments":{"path":"note.txt"}}}"#,
        )
        .await
        .unwrap();
        let text = call["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("hello from hi"), "{text}");
    }
}
