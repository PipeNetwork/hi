//! In-process MCP stdio server that exposes a small tool handler.

use std::io::{self, BufRead, Write};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::{McpError, McpTool};

/// Handles MCP `tools/call` for [`serve_stdio`].
#[async_trait]
pub trait McpStdioHandler: Send {
    fn tools(&self) -> Vec<McpTool>;
    async fn call(&mut self, name: &str, arguments: Value) -> Result<String, String>;
}

/// Serve MCP JSON-RPC on stdin/stdout until EOF.
pub async fn serve_stdio(handler: impl McpStdioHandler) -> Result<u64, McpError> {
    serve_stdio_io(handler, io::BufReader::new(io::stdin()), io::stdout()).await
}

pub async fn serve_stdio_io<R, W, H>(
    mut handler: H,
    reader: R,
    mut writer: W,
) -> Result<u64, McpError>
where
    R: BufRead,
    W: Write,
    H: McpStdioHandler,
{
    let mut handled = 0u64;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = serde_json::from_str(&line)?;
        if message.get("id").is_none() {
            continue;
        }
        handled = handled.saturating_add(1);
        let response = handle_message(&mut handler, &message).await;
        let mut encoded = serde_json::to_vec(&response)?;
        encoded.push(b'\n');
        writer.write_all(&encoded)?;
        writer.flush()?;
    }
    Ok(handled)
}

pub async fn handle_message(handler: &mut impl McpStdioHandler, message: &Value) -> Value {
    let id = message.get("id").cloned().unwrap_or(Value::Null);
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    match method {
        "initialize" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "hi", "version": env!("CARGO_PKG_VERSION") }
            }
        }),
        "tools/list" => {
            let tools = handler.tools();
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "tools": tools }
            })
        }
        "tools/call" => {
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match handler.call(name, arguments).await {
                Ok(text) => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{ "type": "text", "text": text }],
                        "isError": false
                    }
                }),
                Err(err) => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{ "type": "text", "text": err }],
                        "isError": true
                    }
                }),
            }
        }
        "ping" => json!({ "jsonrpc": "2.0", "id": id, "result": {} }),
        other => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32601, "message": format!("method not found: {other}") }
        }),
    }
}

/// Built-in hi tools advertised by `hi mcp serve`.
pub fn hi_serve_tools() -> Vec<McpTool> {
    ["read", "bash", "edit", "write"]
        .into_iter()
        .map(|name| McpTool {
            name: name.into(),
            description: format!("hi {name} tool (sandbox + denylist)"),
            input_schema: json!({ "type": "object" }),
        })
        .collect()
}

/// Round-trip helper for tests: one JSON-RPC request through [`handle_message`].
pub async fn dispatch_line(
    handler: &mut impl McpStdioHandler,
    line: &str,
) -> Result<Value, McpError> {
    let message: Value = serde_json::from_str(line)?;
    Ok(handle_message(handler, &message).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoHandler;

    #[async_trait]
    impl McpStdioHandler for EchoHandler {
        fn tools(&self) -> Vec<McpTool> {
            hi_serve_tools()
        }
        async fn call(&mut self, name: &str, arguments: Value) -> Result<String, String> {
            Ok(format!("{name}:{arguments}"))
        }
    }

    #[tokio::test]
    async fn initialize_and_call_round_trip() {
        let mut handler = EchoHandler;
        let init = dispatch_line(
            &mut handler,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        )
        .await
        .unwrap();
        assert_eq!(init["result"]["serverInfo"]["name"].as_str(), Some("hi"));
        let listed = dispatch_line(
            &mut handler,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        )
        .await
        .unwrap();
        let tools = listed["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 4);
        let call = dispatch_line(
            &mut handler,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"read","arguments":{"path":"a"}}}"#,
        )
        .await
        .unwrap();
        assert_eq!(
            call["result"]["content"][0]["text"].as_str(),
            Some(r#"read:{"path":"a"}"#)
        );
    }
}
