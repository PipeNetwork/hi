//! Merge MCP server configs from `.hi/mcp`, Claude `.mcp.json`, and optional Codex.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::{McpServerConfig, McpTransport, parse_server_config};

/// Where a discovered server config came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McpConfigSource {
    /// Workspace `.hi/mcp/*.json` (wins on name).
    Hi,
    /// Claude / Cursor project `.mcp.json`.
    Claude,
    /// Optional `~/.codex/config.toml` (or `HI_CODEX_CONFIG`).
    Codex,
    /// Auto-attached first-party Pipe `/mcp` (not a workspace file).
    Pipe,
}

impl McpConfigSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hi => "hi",
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Pipe => "pipe",
        }
    }
}

/// Per-source import gating from `hi.toml` `[mcp_import.<source>]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpSourceFilter {
    pub enabled: bool,
    pub only: Vec<String>,
    pub exclude: Vec<String>,
}

impl Default for McpSourceFilter {
    fn default() -> Self {
        Self {
            enabled: true,
            only: Vec::new(),
            exclude: Vec::new(),
        }
    }
}

impl McpSourceFilter {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// `exclude` beats `only`. Source-disabled servers stay visible as blocked.
    pub fn block_reason(&self, name: &str) -> Option<String> {
        if !self.enabled {
            return Some("source disabled".into());
        }
        if self.exclude.iter().any(|item| item == name) {
            return Some("excluded".into());
        }
        if !self.only.is_empty() && !self.only.iter().any(|item| item == name) {
            return Some("not in only list".into());
        }
        None
    }
}

/// Import policy for Claude / Codex / `.hi/mcp`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpImportPolicy {
    pub hi: McpSourceFilter,
    pub claude: McpSourceFilter,
    /// Codex is optional: default off unless `[mcp_import.codex] enabled = true`.
    pub codex: McpSourceFilter,
}

impl Default for McpImportPolicy {
    fn default() -> Self {
        Self {
            hi: McpSourceFilter::default(),
            claude: McpSourceFilter::default(),
            codex: McpSourceFilter::disabled(),
        }
    }
}

/// One discovered server, including blocked/disabled entries.
#[derive(Clone, Debug)]
pub struct DiscoveredMcpServer {
    pub config: McpServerConfig,
    pub source: McpConfigSource,
    pub enabled: bool,
    pub blocked_reason: Option<String>,
}

/// Paths used by [`discover_all_servers_from`].
#[derive(Clone, Debug, Default)]
pub struct McpDiscoveryPaths {
    pub claude_json: Option<PathBuf>,
    pub codex_toml: Option<PathBuf>,
}

/// Project `.mcp.json` plus optional Codex config (exists on disk only).
pub fn default_discovery_paths(project_dir: &Path) -> McpDiscoveryPaths {
    let claude = project_dir.join(".mcp.json");
    McpDiscoveryPaths {
        claude_json: claude.is_file().then_some(claude),
        codex_toml: default_codex_config_path().filter(|path| path.is_file()),
    }
}

/// `HI_CODEX_CONFIG` overrides `~/.codex/config.toml`.
pub fn default_codex_config_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HI_CODEX_CONFIG") {
        let path = PathBuf::from(path);
        return (!path.as_os_str().is_empty()).then_some(path);
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".codex").join("config.toml"))
}

/// Merge sources. Precedence (highest last): Codex, Claude `.mcp.json`, `.hi/mcp`.
pub fn discover_all_servers(
    project_dir: &Path,
    policy: &McpImportPolicy,
) -> Vec<DiscoveredMcpServer> {
    discover_all_servers_from(project_dir, &default_discovery_paths(project_dir), policy)
}

pub fn discover_all_servers_from(
    project_dir: &Path,
    paths: &McpDiscoveryPaths,
    policy: &McpImportPolicy,
) -> Vec<DiscoveredMcpServer> {
    let mut by_name: HashMap<String, DiscoveredMcpServer> = HashMap::new();
    // Lowest priority first so later sources overwrite on name.
    if policy.codex.enabled
        && let Some(path) = &paths.codex_toml
    {
        for config in load_codex_servers(path) {
            insert_discovered(&mut by_name, config, McpConfigSource::Codex, &policy.codex);
        }
    }
    if let Some(path) = &paths.claude_json {
        for config in load_claude_mcp_json(path) {
            insert_discovered(
                &mut by_name,
                config,
                McpConfigSource::Claude,
                &policy.claude,
            );
        }
    }
    for config in crate::discover_hi_servers(project_dir) {
        insert_discovered(&mut by_name, config, McpConfigSource::Hi, &policy.hi);
    }
    let mut out: Vec<_> = by_name.into_values().collect();
    out.sort_by(|a, b| a.config.name.cmp(&b.config.name));
    out
}

fn insert_discovered(
    by_name: &mut HashMap<String, DiscoveredMcpServer>,
    config: McpServerConfig,
    source: McpConfigSource,
    filter: &McpSourceFilter,
) {
    let blocked = filter.block_reason(&config.name);
    let name = config.name.clone();
    by_name.insert(
        name,
        DiscoveredMcpServer {
            enabled: blocked.is_none(),
            blocked_reason: blocked,
            config,
            source,
        },
    );
}

pub fn load_claude_mcp_json(path: &Path) -> Vec<McpServerConfig> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    parse_claude_mcp_json(&text)
}

pub fn parse_claude_mcp_json(text: &str) -> Vec<McpServerConfig> {
    #[derive(serde::Deserialize)]
    struct File {
        #[serde(default, rename = "mcpServers")]
        mcp_servers: HashMap<String, serde_json::Value>,
    }
    let Ok(file) = serde_json::from_str::<File>(text) else {
        return Vec::new();
    };
    let mut configs = Vec::new();
    for (name, value) in file.mcp_servers {
        if let Ok(config) = parse_server_config(&name, &value.to_string()) {
            configs.push(config);
        }
    }
    configs
}

pub fn load_codex_servers(path: &Path) -> Vec<McpServerConfig> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    parse_codex_mcp_toml(&text)
}

pub fn parse_codex_mcp_toml(text: &str) -> Vec<McpServerConfig> {
    let Ok(value) = text.parse::<toml::Value>() else {
        return Vec::new();
    };
    let Some(servers) = value.get("mcp_servers").and_then(|v| v.as_table()) else {
        return Vec::new();
    };
    let mut configs = Vec::new();
    for (name, entry) in servers {
        let Some(table) = entry.as_table() else {
            continue;
        };
        let json = toml_table_to_mcp_json(table);
        if let Ok(config) = parse_server_config(name, &json.to_string()) {
            configs.push(config);
        }
    }
    configs
}

fn toml_table_to_mcp_json(table: &toml::map::Map<String, toml::Value>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    if let Some(command) = table.get("command").and_then(|v| v.as_str()) {
        map.insert("command".into(), serde_json::Value::String(command.into()));
    }
    if let Some(url) = table.get("url").and_then(|v| v.as_str()) {
        map.insert("url".into(), serde_json::Value::String(url.into()));
    }
    if let Some(args) = table.get("args").and_then(|v| v.as_array()) {
        let args: Vec<serde_json::Value> = args
            .iter()
            .filter_map(|v| v.as_str().map(|s| serde_json::Value::String(s.into())))
            .collect();
        map.insert("args".into(), serde_json::Value::Array(args));
    }
    if let Some(env) = table.get("env").and_then(|v| v.as_table()) {
        let mut env_map = serde_json::Map::new();
        for (k, v) in env {
            if let Some(s) = v.as_str() {
                env_map.insert(k.clone(), serde_json::Value::String(s.into()));
            }
        }
        map.insert("env".into(), serde_json::Value::Object(env_map));
    }
    if let Some(headers) = table.get("headers").and_then(|v| v.as_table()) {
        let mut header_map = serde_json::Map::new();
        for (k, v) in headers {
            if let Some(s) = v.as_str() {
                header_map.insert(k.clone(), serde_json::Value::String(s.into()));
            }
        }
        map.insert("headers".into(), serde_json::Value::Object(header_map));
    }
    if let Some(auto) = table.get("auto_reconnect").and_then(|v| v.as_bool()) {
        map.insert("auto_reconnect".into(), serde_json::Value::Bool(auto));
    }
    serde_json::Value::Object(map)
}

/// Expand `${VAR}` (and `$VAR`) in HTTP header values. Unknown vars stay empty.
pub fn expand_env_templates(input: &str) -> String {
    expand_env_templates_with(input, |name| std::env::var(name).unwrap_or_default())
}

fn expand_env_templates_with(input: &str, lookup: impl Fn(&str) -> String) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'{' {
                if let Some(end) = input[i + 2..].find('}') {
                    let name = &input[i + 2..i + 2 + end];
                    if is_env_name(name) {
                        out.push_str(&lookup(name));
                        i += 3 + end;
                        continue;
                    }
                }
            } else {
                let rest = &input[i + 1..];
                let len = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .count();
                if len > 0 {
                    let name = &rest[..len];
                    out.push_str(&lookup(name));
                    i += 1 + len;
                    continue;
                }
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn is_env_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn expand_http_headers(headers: &HashMap<String, String>) -> HashMap<String, String> {
    headers
        .iter()
        .map(|(k, v)| (k.clone(), expand_env_templates(v)))
        .collect()
}

/// Names that must not be silently enabled (blocked by policy).
pub fn blocked_names(discovered: &[DiscoveredMcpServer]) -> HashSet<String> {
    discovered
        .iter()
        .filter(|s| !s.enabled)
        .map(|s| s.config.name.clone())
        .collect()
}

/// Transport label for status tables.
pub fn transport_label(transport: &McpTransport) -> &'static str {
    match transport {
        McpTransport::Stdio { .. } => "stdio",
        McpTransport::Http { .. } => "http",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclude_beats_only() {
        let filter = McpSourceFilter {
            enabled: true,
            only: vec!["keep".into(), "drop".into()],
            exclude: vec!["drop".into()],
        };
        assert_eq!(filter.block_reason("keep"), None);
        assert_eq!(filter.block_reason("drop").as_deref(), Some("excluded"));
        assert_eq!(
            filter.block_reason("other").as_deref(),
            Some("not in only list")
        );
    }

    #[test]
    fn hi_wins_on_name() {
        let tmp = tempfile::tempdir().unwrap();
        let hi = tmp.path().join(".hi").join("mcp");
        std::fs::create_dir_all(&hi).unwrap();
        std::fs::write(hi.join("shared.json"), r#"{"command":"hi-cmd","args":[]}"#).unwrap();
        let claude = tmp.path().join(".mcp.json");
        std::fs::write(
            &claude,
            r#"{"mcpServers":{"shared":{"command":"claude-cmd","args":[]},"only-claude":{"command":"c","args":[]}}}"#,
        )
        .unwrap();
        let found = discover_all_servers_from(
            tmp.path(),
            &McpDiscoveryPaths {
                claude_json: Some(claude),
                codex_toml: None,
            },
            &McpImportPolicy::default(),
        );
        let shared = found.iter().find(|s| s.config.name == "shared").unwrap();
        assert_eq!(shared.source, McpConfigSource::Hi);
        match &shared.config.transport {
            McpTransport::Stdio { command, .. } => assert_eq!(command, "hi-cmd"),
            _ => panic!("stdio"),
        }
        assert!(found.iter().any(|s| s.config.name == "only-claude"));
    }

    #[test]
    fn blocked_name_stays_visible_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        let claude = tmp.path().join(".mcp.json");
        std::fs::write(
            &claude,
            r#"{"mcpServers":{"blocked":{"command":"x","args":[]},"ok":{"command":"y","args":[]}}}"#,
        )
        .unwrap();
        let mut policy = McpImportPolicy::default();
        policy.claude.exclude.push("blocked".into());
        let found = discover_all_servers_from(
            tmp.path(),
            &McpDiscoveryPaths {
                claude_json: Some(claude),
                codex_toml: None,
            },
            &policy,
        );
        let blocked = found.iter().find(|s| s.config.name == "blocked").unwrap();
        assert!(!blocked.enabled);
        assert_eq!(blocked.blocked_reason.as_deref(), Some("excluded"));
        let ok = found.iter().find(|s| s.config.name == "ok").unwrap();
        assert!(ok.enabled);
    }

    #[test]
    fn parse_codex_toml_stdio() {
        let text = r#"
[mcp_servers.docs]
command = "npx"
args = ["-y", "docs"]

[mcp_servers.docs.env]
TOKEN = "x"
"#;
        let configs = parse_codex_mcp_toml(text);
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "docs");
        match &configs[0].transport {
            McpTransport::Stdio { command, args, env } => {
                assert_eq!(command, "npx");
                assert_eq!(args, &["-y".to_string(), "docs".to_string()]);
                assert_eq!(env.get("TOKEN").map(String::as_str), Some("x"));
            }
            _ => panic!("stdio"),
        }
    }

    #[test]
    fn expand_dollar_braces() {
        assert_eq!(
            expand_env_templates_with("Bearer ${HI_MCP_EXPAND_TEST}", |name| {
                if name == "HI_MCP_EXPAND_TEST" {
                    "secret".into()
                } else {
                    String::new()
                }
            }),
            "Bearer secret"
        );
        assert_eq!(
            expand_env_templates_with("token=$TOKEN extra", |name| {
                if name == "TOKEN" {
                    "abc".into()
                } else {
                    String::new()
                }
            }),
            "token=abc extra"
        );
    }
}
