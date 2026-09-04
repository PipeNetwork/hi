//! Connect workspace MCP servers and expose them through search/select tools.
//!
//! Each MCP tool's JSON Schema stays off the model request. The agent sees
//! two gateway tools (`search_tool`, `use_tool`); `search_tool` returns names
//! and schemas on demand. Servers are registered at startup without waiting
//! on handshakes; the first `use_tool` connects with a short grace period.
//!
//! First-party Pipe `/mcp` is auto-attached as server `pipe` when the provider
//! has an `mcp_url` and API key. Folder trust still gates repo-local stdio
//! servers; remote Pipe does not require trust.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use hi_mcp::{
    AgentToolPolicy, DiscoveredMcpServer, McpAdminCmd, McpConfigSource, McpImportPolicy,
    McpServerConfig, PIPE_SERVER_NAME, ServerAllowList, parse_mcp_admin,
};
use hi_tools::{McpBackend, McpToolInfo};
use serde_json::Value;

pub struct ConnectedMcp {
    client: tokio::sync::Mutex<hi_mcp::McpClient>,
    workspace_root: PathBuf,
    trusted: bool,
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

    async fn read_resource(&self, server: &str, uri: &str) -> Result<String> {
        self.client
            .lock()
            .await
            .read_resource(server, uri)
            .await
            .map_err(|error| anyhow::anyhow!("{error}"))
    }

    async fn workspace_status(&self) -> String {
        self.client.lock().await.status_table()
    }

    async fn workspace_admin(&self, args: &str) -> Result<String> {
        let cmd = parse_mcp_admin(args);
        if matches!(cmd, McpAdminCmd::AddStdio { .. }) && !self.trusted {
            anyhow::bail!(
                "folder is untrusted; stdio MCP add is blocked. Use --http or /trust on."
            );
        }
        let mut client = self.client.lock().await;
        let text = client
            .admin(args)
            .await
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        let extra = match cmd {
            McpAdminCmd::Allow { server, .. } | McpAdminCmd::Deny { server, .. } => {
                persist_toml_policy_if_needed(&self.workspace_root, &client, &server)?
            }
            _ => String::new(),
        };
        Ok(format!("{text}{extra}"))
    }
}

impl ConnectedMcp {
    pub async fn test_server(&self, name: &str) -> Result<String> {
        let mut client = self.client.lock().await;
        client
            .test_server(name)
            .await
            .map_err(|err| anyhow::anyhow!("{err}"))
    }
}

/// Credentials and extra allow names for auto-attaching first-party Pipe MCP.
#[derive(Clone, Debug)]
pub struct PipeAttach {
    pub url: String,
    pub api_key: String,
    pub extra_allow: Vec<String>,
}

/// Why first-party Pipe was not requested for auto-attach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipeAttachSkip {
    Disabled,
    MissingUrl,
    MissingKey,
}

/// Outcome of auto-attach after workspace discovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipeAttachStatus {
    Attached,
    Disabled,
    MissingUrl,
    MissingKey,
    NameCollision,
    NotRequested,
}

impl PipeAttachStatus {
    pub fn doctor_check(self) -> hi_agent::doctor::Check {
        use hi_agent::doctor::Check;
        match self {
            Self::Attached => Check::pass(
                "mcp agent attach",
                "pipe (allowlist: pipe.models.list, pipe.models.health)",
            ),
            Self::Disabled => Check::pass("mcp agent attach", "off ([mcp.pipe] enabled = false)"),
            Self::MissingUrl => {
                Check::pass("mcp agent attach", "skipped: no mcp_url for this provider")
            }
            Self::MissingKey => Check::fail(
                "mcp agent attach",
                "mcp_url is set but no API key",
                "set a project API key, or disable with [mcp.pipe] enabled = false",
            ),
            Self::NameCollision => Check::pass(
                "mcp agent attach",
                "skipped: workspace already defines a server named 'pipe'",
            ),
            Self::NotRequested => Check::pass("mcp agent attach", "not requested"),
        }
    }
}

pub fn decide_pipe_attach(
    enabled: bool,
    mcp_url: Option<&str>,
    api_key: &str,
    extra_allow: Vec<String>,
) -> Result<PipeAttach, PipeAttachSkip> {
    if !enabled {
        return Err(PipeAttachSkip::Disabled);
    }
    let Some(url) = mcp_url.map(str::trim).filter(|url| !url.is_empty()) else {
        return Err(PipeAttachSkip::MissingUrl);
    };
    if api_key.trim().is_empty() {
        return Err(PipeAttachSkip::MissingKey);
    }
    Ok(PipeAttach {
        url: url.to_string(),
        api_key: api_key.to_string(),
        extra_allow,
    })
}

fn persist_toml_policy_if_needed(
    workspace_root: &Path,
    client: &hi_mcp::McpClient,
    server: &str,
) -> Result<String> {
    let json_path = hi_mcp::hi_mcp_json_path(workspace_root, server);
    if json_path.is_file() {
        return Ok(String::new());
    }
    let list = client.policy().server_list(server);
    let pipe_extra = (server == PIPE_SERVER_NAME).then(|| client.policy().pipe_extra_allow());
    let path = persist_toml_server_policy(workspace_root, server, &list, pipe_extra)?;
    Ok(format!(
        "\nsaved [mcp.servers.{server}] in {}",
        path.display()
    ))
}

fn persist_toml_server_policy(
    workspace_root: &Path,
    name: &str,
    list: &ServerAllowList,
    pipe_extra: Option<&[String]>,
) -> Result<std::path::PathBuf> {
    let path = workspace_root.join("hi.toml");
    let mut config: crate::config::Config = if path.is_file() {
        toml::from_str(&std::fs::read_to_string(&path)?)?
    } else {
        crate::config::Config::default()
    };
    if list.only.is_empty() && list.exclude.is_empty() {
        config.mcp.servers.remove(name);
    } else {
        config.mcp.servers.insert(
            name.to_string(),
            crate::config::McpServerPolicySection {
                only: list.only.clone(),
                exclude: list.exclude.clone(),
            },
        );
    }
    if let Some(extra) = pipe_extra {
        config.mcp.pipe.allow = extra.to_vec();
    }
    crate::config::save_config_to(&config, &path)?;
    Ok(path)
}

fn first_party_pipe_server(attach: &PipeAttach) -> DiscoveredMcpServer {
    let config = McpServerConfig::http(PIPE_SERVER_NAME, attach.url.clone())
        .with_header("Authorization", format!("Bearer {}", attach.api_key));
    DiscoveredMcpServer {
        config,
        source: McpConfigSource::Pipe,
        enabled: true,
        blocked_reason: None,
    }
}

pub async fn connect_workspace_mcp(
    workspace_root: &Path,
    policy: &McpImportPolicy,
    pipe: Option<&PipeAttach>,
) -> (Option<Arc<ConnectedMcp>>, PipeAttachStatus) {
    connect_workspace_mcp_with_policies(workspace_root, policy, pipe, &HashMap::new()).await
}

/// Discover merged MCP sources and register them without blocking on connect.
/// Fail-open: a dead server never prevents the TUI from starting.
///
/// Repo-local servers require folder trust. First-party Pipe HTTP does not.
pub async fn connect_workspace_mcp_with_policies(
    workspace_root: &Path,
    policy: &McpImportPolicy,
    pipe: Option<&PipeAttach>,
    server_policies: &HashMap<String, ServerAllowList>,
) -> (Option<Arc<ConnectedMcp>>, PipeAttachStatus) {
    let trusted = matches!(
        hi_tools::folder_trust::resolve_trust(workspace_root),
        hi_tools::folder_trust::TrustOutcome::Trusted
    );
    connect_workspace_mcp_with_trust(workspace_root, policy, pipe, trusted, server_policies).await
}

pub async fn connect_workspace_mcp_with_trust(
    workspace_root: &Path,
    policy: &McpImportPolicy,
    pipe: Option<&PipeAttach>,
    workspace_trusted: bool,
    server_policies: &HashMap<String, ServerAllowList>,
) -> (Option<Arc<ConnectedMcp>>, PipeAttachStatus) {
    let mut discovered = if workspace_trusted {
        hi_mcp::discover_all_servers(workspace_root, policy)
    } else {
        Vec::new()
    };
    let collision = discovered
        .iter()
        .any(|server| server.config.name == PIPE_SERVER_NAME);
    let pipe_status = match (pipe, collision) {
        (None, _) => PipeAttachStatus::NotRequested,
        (Some(_), true) => {
            eprintln!(
                "mcp: skip auto-attach {PIPE_SERVER_NAME}: workspace already defines that name"
            );
            PipeAttachStatus::NameCollision
        }
        (Some(attach), false) => {
            discovered.push(first_party_pipe_server(attach));
            PipeAttachStatus::Attached
        }
    };
    let Some(runner) = hi_tools::ProcessRunner::new(workspace_root).ok() else {
        return (None, pipe_status);
    };
    let mut client = hi_mcp::McpClient::with_process_runner(runner);
    if let Some(attach) = pipe {
        client.set_agent_tool_policy(AgentToolPolicy::with_pipe_extra_allow(
            attach.extra_allow.clone(),
        ));
    }
    client.register_all(discovered);
    client.overlay_server_policies(
        server_policies
            .iter()
            .map(|(name, list)| (name.as_str(), list)),
    );
    (
        Some(Arc::new(ConnectedMcp {
            client: tokio::sync::Mutex::new(client),
            workspace_root: workspace_root.to_path_buf(),
            trusted: workspace_trusted,
        })),
        pipe_status,
    )
}

/// Eager connect used by `hi mcp test` (times the handshake).
pub async fn test_workspace_mcp(
    workspace_root: &Path,
    policy: &McpImportPolicy,
    pipe: Option<&PipeAttach>,
    name: &str,
) -> Result<String> {
    let (Some(host), _) = connect_workspace_mcp(workspace_root, policy, pipe).await else {
        anyhow::bail!("no workspace MCP servers (and no first-party pipe attach)");
    };
    host.test_server(name).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

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
            workspace_root: std::env::temp_dir(),
            trusted: true,
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
    async fn empty_workspace_still_has_host_for_mcp_add() {
        let tmp = tempfile::tempdir().unwrap();
        let (host, status) = connect_workspace_mcp_with_trust(
            tmp.path(),
            &McpImportPolicy::default(),
            None,
            true,
            &HashMap::new(),
        )
        .await;
        assert_eq!(status, PipeAttachStatus::NotRequested);
        let host = host.expect("empty host");
        let added = host
            .workspace_admin("add docs --http https://example.test/mcp")
            .await
            .unwrap();
        assert!(added.contains("added 'docs'"), "{added}");
        assert!(hi_mcp::hi_mcp_json_path(tmp.path(), "docs").is_file());
    }

    #[tokio::test]
    async fn lazy_connect_does_not_wait_at_startup() {
        let tmp = tempfile::tempdir().unwrap();
        let mcp_dir = tmp.path().join(".hi").join("mcp");
        std::fs::create_dir_all(&mcp_dir).unwrap();
        std::fs::write(
            mcp_dir.join("slow.json"),
            r#"{"command":"sleep","args":["30"]}"#,
        )
        .unwrap();
        let started = std::time::Instant::now();
        // This test isolates lazy registration from folder-trust admission.
        // Repo-local MCP is intentionally unavailable in an untrusted,
        // headless workspace, so grant trust explicitly through the internal
        // test seam instead of weakening the production trust resolver.
        let (host, _) = connect_workspace_mcp_with_trust(
            tmp.path(),
            &McpImportPolicy::default(),
            None,
            true,
            &HashMap::new(),
        )
        .await;
        let host = host.expect("registered");
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "startup blocked on MCP connect: {elapsed:?}"
        );
        let status = host.workspace_status().await;
        assert!(status.contains("slow"), "{status}");
        assert!(
            status.contains("registered")
                || status.contains("disconnected")
                || status.contains("failed"),
            "{status}"
        );
    }

    #[tokio::test]
    async fn failed_server_returns_actionable_call_error() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = hi_tools::ProcessRunner::new(tmp.path()).unwrap();
        let mut client = hi_mcp::McpClient::with_process_runner(runner);
        client.register(DiscoveredMcpServer {
            config: hi_mcp::McpServerConfig::stdio("dead", "false", &[]),
            source: hi_mcp::McpConfigSource::Hi,
            enabled: true,
            blocked_reason: None,
        });
        let backend = ConnectedMcp {
            client: tokio::sync::Mutex::new(client),
            workspace_root: std::env::temp_dir(),
            trusted: true,
        };
        let err = backend.call("dead", "echo", &json!({})).await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("reconnect") || msg.contains("not connected"),
            "{msg}"
        );
    }

    #[tokio::test]
    async fn auto_attach_pipe_without_workspace_mcp() {
        let tmp = tempfile::tempdir().unwrap();
        let attach = PipeAttach {
            url: "http://127.0.0.1:9/mcp".into(),
            api_key: "sk-test".into(),
            extra_allow: Vec::new(),
        };
        let (host, status) = connect_workspace_mcp_with_trust(
            tmp.path(),
            &McpImportPolicy::default(),
            Some(&attach),
            false,
            &HashMap::new(),
        )
        .await;
        assert_eq!(status, PipeAttachStatus::Attached);
        let table = host.expect("pipe registered").workspace_status().await;
        assert!(
            table.lines().any(|line| {
                let mut parts = line.split_whitespace();
                parts.next() == Some("pipe") && parts.next() == Some("pipe")
            }),
            "expected first-party source column 'pipe': {table}"
        );
    }

    #[tokio::test]
    async fn untrusted_folder_skips_stdio_but_attaches_pipe() {
        let tmp = tempfile::tempdir().unwrap();
        let mcp_dir = tmp.path().join(".hi").join("mcp");
        std::fs::create_dir_all(&mcp_dir).unwrap();
        std::fs::write(
            mcp_dir.join("local.json"),
            r#"{"command":"false","args":[]}"#,
        )
        .unwrap();
        let attach = PipeAttach {
            url: "http://127.0.0.1:9/mcp".into(),
            api_key: "sk-test".into(),
            extra_allow: Vec::new(),
        };
        let (host, status) = connect_workspace_mcp_with_trust(
            tmp.path(),
            &McpImportPolicy::default(),
            Some(&attach),
            false,
            &HashMap::new(),
        )
        .await;
        assert_eq!(status, PipeAttachStatus::Attached);
        let table = host.expect("pipe").workspace_status().await;
        assert!(table.contains(PIPE_SERVER_NAME), "{table}");
        assert!(!table.contains("local"), "{table}");
    }

    #[tokio::test]
    async fn workspace_pipe_json_wins_over_auto_attach() {
        let tmp = tempfile::tempdir().unwrap();
        let mcp_dir = tmp.path().join(".hi").join("mcp");
        std::fs::create_dir_all(&mcp_dir).unwrap();
        std::fs::write(
            mcp_dir.join("pipe.json"),
            r#"{"command":"false","args":[]}"#,
        )
        .unwrap();
        let attach = PipeAttach {
            url: "http://127.0.0.1:9/mcp".into(),
            api_key: "sk-test".into(),
            extra_allow: Vec::new(),
        };
        let (host, status) = connect_workspace_mcp_with_trust(
            tmp.path(),
            &McpImportPolicy::default(),
            Some(&attach),
            true,
            &HashMap::new(),
        )
        .await;
        assert_eq!(status, PipeAttachStatus::NameCollision);
        let table = host.expect("workspace pipe").workspace_status().await;
        assert!(table.contains(PIPE_SERVER_NAME), "{table}");
        assert!(
            table.lines().any(|line| {
                let mut parts = line.split_whitespace();
                parts.next() == Some("pipe") && parts.next() == Some("hi")
            }),
            "workspace pipe.json should keep source hi: {table}"
        );
    }

    #[test]
    fn decide_pipe_attach_reasons() {
        assert!(matches!(
            decide_pipe_attach(false, Some("https://api.pipenetwork.ai/mcp"), "k", vec![]),
            Err(PipeAttachSkip::Disabled)
        ));
        assert!(matches!(
            decide_pipe_attach(true, None, "k", vec![]),
            Err(PipeAttachSkip::MissingUrl)
        ));
        assert!(matches!(
            decide_pipe_attach(true, Some("https://x/mcp"), "", vec![]),
            Err(PipeAttachSkip::MissingKey)
        ));
        let ok = decide_pipe_attach(
            true,
            Some("https://x/mcp"),
            "k",
            vec!["pipe.usage.summary".into()],
        )
        .unwrap();
        assert_eq!(ok.url, "https://x/mcp");
        assert_eq!(ok.extra_allow.len(), 1);
    }

    #[tokio::test]
    async fn deny_imported_server_persists_hi_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = hi_tools::ProcessRunner::new(tmp.path()).unwrap();
        let mut client = hi_mcp::McpClient::with_process_runner(runner);
        client.register(DiscoveredMcpServer {
            config: hi_mcp::McpServerConfig::http("docs", "https://example.test/mcp"),
            source: hi_mcp::McpConfigSource::Claude,
            enabled: true,
            blocked_reason: None,
        });
        let backend = ConnectedMcp {
            client: tokio::sync::Mutex::new(client),
            workspace_root: tmp.path().to_path_buf(),
            trusted: true,
        };
        let out = backend.workspace_admin("docs deny wipe").await.unwrap();
        assert!(out.contains("denied 'wipe'"), "{out}");
        let text = std::fs::read_to_string(tmp.path().join("hi.toml")).unwrap();
        assert!(text.contains("[mcp.servers.docs]"), "{text}");
        assert!(text.contains("wipe"), "{text}");
    }

    #[test]
    fn doctor_check_explains_attach_outcomes() {
        let attached = PipeAttachStatus::Attached.doctor_check();
        assert!(attached.passed);
        assert!(
            attached
                .detail
                .as_deref()
                .unwrap()
                .contains("pipe.models.list")
        );
        let collision = PipeAttachStatus::NameCollision.doctor_check();
        assert!(collision.passed);
        assert!(
            collision
                .detail
                .as_deref()
                .unwrap()
                .contains("already defines")
        );
        let missing_key = PipeAttachStatus::MissingKey.doctor_check();
        assert!(!missing_key.passed);
        let disabled = PipeAttachStatus::Disabled.doctor_check();
        assert!(disabled.passed);
        assert!(
            disabled
                .detail
                .as_deref()
                .unwrap()
                .contains("enabled = false")
        );
    }
}
