//! Streamable HTTP JSON-RPC transport (MCP 2025-06-18 / 2024-11-05+).

use std::collections::HashMap;

use async_trait::async_trait;
use reqwest::StatusCode;

use crate::discover::expand_http_headers;
use crate::{McpError, McpTransportTrait};

pub struct HttpTransport {
    client: reqwest::Client,
    url: String,
    headers: HashMap<String, String>,
    session_id: Option<String>,
    next_id: u64,
}

impl HttpTransport {
    pub fn connect(
        url: impl Into<String>,
        headers: HashMap<String, String>,
    ) -> Result<Self, McpError> {
        let headers = expand_http_headers(&headers);
        // Same identity as PipeMcpClient so Pipe's trusted-hi discovery caps apply.
        let client = hi_ai::agent_http_client_quick();
        Ok(Self {
            client,
            url: url.into(),
            headers,
            session_id: None,
            next_id: 1,
        })
    }

    async fn post(
        &mut self,
        body: &serde_json::Value,
    ) -> Result<(StatusCode, Option<String>, String), McpError> {
        let mut request = self.client.post(&self.url).json(body);
        request = request.header("Content-Type", "application/json");
        request = request.header("Accept", "application/json, text/event-stream");
        for (key, value) in &self.headers {
            request = request.header(key.as_str(), value.as_str());
        }
        if let Some(session) = &self.session_id {
            request = request.header("Mcp-Session-Id", session.as_str());
        }
        let response = request
            .send()
            .await
            .map_err(|err| McpError::Transport(err.to_string()))?;
        let status = response.status();
        let session = response
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let text = response
            .text()
            .await
            .map_err(|err| McpError::Transport(err.to_string()))?;
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            return Err(McpError::Auth(format!("HTTP {status}: {text}")));
        }
        if !status.is_success() {
            return Err(McpError::Transport(format!("HTTP {status}: {text}")));
        }
        if let Some(session) = session {
            self.session_id = Some(session);
        }
        let body = if content_type.contains("text/event-stream") {
            sse_json_payload(&text).ok_or_else(|| {
                McpError::Transport("HTTP MCP SSE response had no JSON data frame".into())
            })?
        } else {
            text
        };
        Ok((status, self.session_id.clone(), body))
    }
}

fn sse_json_payload(text: &str) -> Option<String> {
    let mut data = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
        } else if line.is_empty() && !data.is_empty() {
            break;
        }
    }
    if data.is_empty() { None } else { Some(data) }
}

fn parse_rpc_result(id: u64, body: &str) -> Result<serde_json::Value, McpError> {
    let message: serde_json::Value = serde_json::from_str(body)?;
    if let Some(error) = message.get("error") {
        return Err(McpError::Server(error.to_string()));
    }
    if let Some(response_id) = message.get("id")
        && !response_id.is_null()
        && response_id.as_u64() != Some(id)
        && response_id.as_str() != Some(&id.to_string())
    {
        return Err(McpError::ResponseIdMismatch {
            expected: id,
            actual: response_id.to_string(),
        });
    }
    Ok(message
        .get("result")
        .cloned()
        .unwrap_or(serde_json::Value::Null))
}

#[async_trait]
impl McpTransportTrait for HttpTransport {
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
        let (_, _, body) = self.post(&request).await?;
        parse_rpc_result(id, &body)
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
        // Pipe returns method_not_found for notifications/initialized. JSON-RPC
        // and HTTP errors on notifications must not fail the handshake.
        let _ = self.post(&notification).await;
        Ok(())
    }

    async fn close(&mut self) -> Result<(), McpError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{McpClient, McpServerConfig};
    use std::sync::atomic::{AtomicU16, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    static PORT_HINT: AtomicU16 = AtomicU16::new(0);

    async fn spawn_json_server(
        handler: fn(&str) -> (u16, &'static str, String),
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let _ = PORT_HINT.fetch_add(1, Ordering::Relaxed);
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 65536];
            let n = stream.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..n]);
            let (status, content_type, body) = handler(&request);
            let response = format!(
                "HTTP/1.1 {status} OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
        });
        (format!("http://{addr}/mcp"), handle)
    }

    fn rpc_ok(method_substr: &str, request: &str, result: serde_json::Value) -> String {
        let id = if request.contains(method_substr) {
            // crude: first "id":N
            request
                .split("\"id\":")
                .nth(1)
                .and_then(|s| {
                    s.chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect::<String>()
                        .parse::<u64>()
                        .ok()
                })
                .unwrap_or(1)
        } else {
            1
        };
        serde_json::json!({"jsonrpc":"2.0","id":id,"result":result}).to_string()
    }

    #[tokio::test]
    async fn http_initialize_list_and_call() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    let n = match stream.read(&mut buf).await {
                        Ok(n) if n > 0 => n,
                        _ => return,
                    };
                    let request = String::from_utf8_lossy(&buf[..n]);
                    let body = if request.contains("initialize") {
                        rpc_ok(
                            "initialize",
                            &request,
                            serde_json::json!({
                                "protocolVersion":"2024-11-05",
                                "capabilities":{"tools":{}},
                                "serverInfo":{"name":"fake-http","version":"1"}
                            }),
                        )
                    } else if request.contains("tools/list") {
                        rpc_ok(
                            "tools/list",
                            &request,
                            serde_json::json!({"tools":[{
                                "name":"echo",
                                "description":"echo",
                                "inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}
                            }]}),
                        )
                    } else if request.contains("tools/call") {
                        rpc_ok(
                            "tools/call",
                            &request,
                            serde_json::json!({"content":[{"type":"text","text":"pong"}]}),
                        )
                    } else {
                        rpc_ok("notifications", &request, serde_json::json!({}))
                    };
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        let url = format!("http://{addr}/mcp");
        let mut client = McpClient::new();
        client
            .connect(McpServerConfig::http("http-demo", url))
            .await
            .unwrap();
        let tools = client.list_tools("http-demo").unwrap();
        assert_eq!(tools[0].name, "echo");
        let result = client
            .invoke_tool("http-demo", "echo", serde_json::json!({"text":"hi"}))
            .await
            .unwrap();
        assert_eq!(result.content, "pong");
    }

    #[tokio::test]
    async fn http_401_is_auth() {
        let (url, _handle) =
            spawn_json_server(|_| (401, "application/json", r#"{"error":"nope"}"#.into())).await;
        let mut client = McpClient::new();
        let err = client
            .connect(McpServerConfig::http("auth", url))
            .await
            .unwrap_err();
        assert!(
            matches!(err, McpError::Auth(_)),
            "expected Auth, got {err:?}"
        );
    }

    #[tokio::test]
    async fn http_sends_hi_agent_identity() {
        let (tx, rx) = tokio::sync::oneshot::channel::<String>();
        let tx = std::sync::Mutex::new(Some(tx));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 65536];
            let n = stream.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).into_owned();
            if let Some(tx) = tx.lock().unwrap().take() {
                let _ = tx.send(request);
            }
            let body = r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"id","version":"1"}}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
        });
        let mut client = McpClient::new();
        let _ = client
            .connect(McpServerConfig::http("id", format!("http://{addr}/mcp")))
            .await;
        let headers = rx.await.unwrap().to_ascii_lowercase();
        assert!(
            headers.contains("ai_agent: hi"),
            "missing AI_AGENT identity: {headers}"
        );
        assert!(
            headers.contains("user-agent: hi/"),
            "missing hi/ User-Agent: {headers}"
        );
    }

    #[tokio::test]
    async fn initialized_notification_jsonrpc_error_still_connects() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    let n = match stream.read(&mut buf).await {
                        Ok(n) if n > 0 => n,
                        _ => return,
                    };
                    let request = String::from_utf8_lossy(&buf[..n]);
                    let body = if request.contains("notifications/initialized") {
                        r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"Method not found"}}"#
                            .to_string()
                    } else if request.contains("initialize") {
                        rpc_ok(
                            "initialize",
                            &request,
                            serde_json::json!({
                                "protocolVersion":"2025-06-18",
                                "capabilities":{"tools":{}},
                                "serverInfo":{"name":"pipe-mcp","version":"1"}
                            }),
                        )
                    } else if request.contains("tools/list") {
                        rpc_ok("tools/list", &request, serde_json::json!({"tools":[]}))
                    } else {
                        rpc_ok("other", &request, serde_json::json!({}))
                    };
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        let mut client = McpClient::new();
        client
            .connect(McpServerConfig::http(
                "notify-demo",
                format!("http://{addr}/mcp"),
            ))
            .await
            .unwrap();
        assert_eq!(
            client.status("notify-demo").unwrap(),
            crate::ServerStatus::Connected
        );
    }

    fn pipe_tools_list() -> serde_json::Value {
        serde_json::json!({
            "tools": [
                {"name":"pipe.models.list","description":"list","inputSchema":{"type":"object"}},
                {"name":"pipe.models.health","description":"health","inputSchema":{"type":"object"}},
                {"name":"pipe.chat.completions.create","description":"chat","inputSchema":{"type":"object"}},
                {"name":"pipe.responses.create","description":"responses","inputSchema":{"type":"object"}},
                {"name":"pipe.usage.summary","description":"usage","inputSchema":{"type":"object"}},
                {"name":"pipe.request.get","description":"get","inputSchema":{"type":"object"}}
            ]
        })
    }

    async fn spawn_pipe_shaped_http() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    let n = match stream.read(&mut buf).await {
                        Ok(n) if n > 0 => n,
                        _ => return,
                    };
                    let request = String::from_utf8_lossy(&buf[..n]);
                    let body = if request.contains("initialize") && !request.contains("initialized")
                    {
                        rpc_ok(
                            "initialize",
                            &request,
                            serde_json::json!({
                                "protocolVersion":"2025-06-18",
                                "capabilities":{"tools":{}},
                                "serverInfo":{"name":"pipe-mcp","version":"1"}
                            }),
                        )
                    } else if request.contains("tools/list") {
                        rpc_ok("tools/list", &request, pipe_tools_list())
                    } else if request.contains("tools/call") {
                        rpc_ok(
                            "tools/call",
                            &request,
                            serde_json::json!({"content":[{"type":"text","text":"{\"models\":[]}"}]}),
                        )
                    } else if request.contains("notifications/initialized") {
                        r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"Method not found"}}"#
                            .to_string()
                    } else {
                        rpc_ok("other", &request, serde_json::json!({}))
                    };
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}/mcp")
    }

    #[tokio::test]
    async fn pipe_allowlist_filters_search_and_blocks_nested_chat() {
        use crate::{AgentToolPolicy, DiscoveredMcpServer, McpConfigSource, PIPE_SERVER_NAME};
        let url = spawn_pipe_shaped_http().await;
        let mut client = McpClient::new();
        client.register(DiscoveredMcpServer {
            config: McpServerConfig::http(PIPE_SERVER_NAME, url),
            source: McpConfigSource::Pipe,
            enabled: true,
            blocked_reason: None,
        });
        client
            .ensure_connected(PIPE_SERVER_NAME, std::time::Duration::from_secs(8))
            .await
            .unwrap();
        let tools = client.list_tools(PIPE_SERVER_NAME).unwrap();
        let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["pipe.models.list", "pipe.models.health"]);
        client
            .invoke_tool(PIPE_SERVER_NAME, "pipe.models.list", serde_json::json!({}))
            .await
            .unwrap();
        let err = client
            .invoke_tool(
                PIPE_SERVER_NAME,
                "pipe.chat.completions.create",
                serde_json::json!({}),
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("not agent-callable")
                || err.to_string().contains("nested model"),
            "{err}"
        );

        client.set_agent_tool_policy(AgentToolPolicy::with_pipe_extra_allow([
            "pipe.usage.summary",
        ]));
        let tools = client.list_tools(PIPE_SERVER_NAME).unwrap();
        assert!(tools.iter().any(|t| t.name == "pipe.usage.summary"));
        assert!(
            !tools
                .iter()
                .any(|t| t.name == "pipe.chat.completions.create")
        );
        assert!(!tools.iter().any(|t| t.name == "pipe.responses.create"));
        client
            .invoke_tool(
                PIPE_SERVER_NAME,
                "pipe.usage.summary",
                serde_json::json!({}),
            )
            .await
            .unwrap();
        let err = client
            .invoke_tool(
                PIPE_SERVER_NAME,
                "pipe.responses.create",
                serde_json::json!({}),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not agent-callable"), "{err}");
    }
}
