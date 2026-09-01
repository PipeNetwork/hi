//! Agent-facing MCP tool policy.
//!
//! First-party Pipe stays on a fail-closed default list (nested chat/responses
//! are never agent-callable). Imported servers (`.hi/mcp`, Claude, Codex) are
//! visible by default; optional per-server `only` / `exclude` lists hide tools
//! the same way Pipe's allowlist does.

use std::collections::HashMap;

use crate::McpConfigSource;

/// Reserved name for the auto-attached Pipe MCP server.
pub const PIPE_SERVER_NAME: &str = "pipe";

/// Default agent-callable first-party tools.
pub const PIPE_DEFAULT_ALLOWED_TOOLS: &[&str] = &["pipe.models.list", "pipe.models.health"];

/// Tools that must never be agent-callable, even if listed in `[mcp.pipe] allow`.
pub const PIPE_NEVER_AGENT_TOOLS: &[&str] =
    &["pipe.chat.completions.create", "pipe.responses.create"];

/// Per-server `only` / `exclude` (JSON or `[mcp.servers.<name>]`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServerAllowList {
    /// When non-empty, only these tools are visible (Pipe still unions defaults
    /// and extra-allow).
    pub only: Vec<String>,
    /// Always hidden. Wins over `only` and Pipe extras.
    pub exclude: Vec<String>,
}

/// Per-session extra allows plus optional per-server lists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AgentToolPolicy {
    extra_allow: Vec<String>,
    servers: HashMap<String, ServerAllowList>,
}

impl AgentToolPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Extra tool names allowed on the first-party `pipe` server.
    /// Nested chat/responses names are ignored.
    pub fn with_pipe_extra_allow(extra: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            extra_allow: extra
                .into_iter()
                .map(Into::into)
                .filter(|name| !is_never_agent_pipe_tool(name))
                .collect(),
            servers: HashMap::new(),
        }
    }

    /// Pipe extras persisted to `[mcp.pipe] allow`.
    pub fn pipe_extra_allow(&self) -> &[String] {
        &self.extra_allow
    }

    pub fn server_list(&self, name: &str) -> ServerAllowList {
        self.servers.get(name).cloned().unwrap_or_default()
    }

    /// Replace lists from `.hi/mcp/<name>.json` at register time.
    pub fn set_from_config(&mut self, server: &str, only: &[String], exclude: &[String]) {
        if only.is_empty() && exclude.is_empty() {
            return;
        }
        let entry = self.servers.entry(server.to_string()).or_default();
        if !only.is_empty() {
            entry.only = only.to_vec();
        }
        if !exclude.is_empty() {
            entry.exclude = exclude.to_vec();
        }
    }

    /// Overlay `[mcp.servers.<name>]`: non-empty `only` replaces; `exclude` unions.
    pub fn overlay(&mut self, server: &str, overlay: &ServerAllowList) {
        if overlay.only.is_empty() && overlay.exclude.is_empty() {
            return;
        }
        let entry = self.servers.entry(server.to_string()).or_default();
        if !overlay.only.is_empty() {
            entry.only = overlay.only.clone();
        }
        for tool in &overlay.exclude {
            if !entry.exclude.iter().any(|existing| existing == tool) {
                entry.exclude.push(tool.clone());
            }
        }
    }

    pub fn allows(&self, source: McpConfigSource, server: &str, tool: &str) -> bool {
        if source == McpConfigSource::Pipe && is_never_agent_pipe_tool(tool) {
            return false;
        }
        let list = self.servers.get(server);
        if let Some(list) = list
            && list.exclude.iter().any(|denied| denied == tool)
        {
            return false;
        }
        if source == McpConfigSource::Pipe {
            return PIPE_DEFAULT_ALLOWED_TOOLS.contains(&tool)
                || self.extra_allow.iter().any(|allowed| allowed == tool)
                || list.is_some_and(|list| {
                    !list.only.is_empty() && list.only.iter().any(|allowed| allowed == tool)
                });
        }
        if let Some(list) = list
            && !list.only.is_empty()
        {
            return list.only.iter().any(|allowed| allowed == tool);
        }
        true
    }

    pub fn deny_reason(&self, source: McpConfigSource, server: &str, tool: &str) -> Option<String> {
        if self.allows(source, server, tool) {
            return None;
        }
        if source == McpConfigSource::Pipe && is_never_agent_pipe_tool(tool) {
            return Some(format!(
                "tool '{tool}' is not agent-callable on server '{server}' \
                 (nested model calls stay off the coding loop)"
            ));
        }
        if let Some(list) = self.servers.get(server) {
            if list.exclude.iter().any(|denied| denied == tool) {
                return Some(format!("tool '{tool}' is excluded on server '{server}'"));
            }
            if !list.only.is_empty() {
                return Some(format!(
                    "tool '{tool}' is not on the only-list for server '{server}'"
                ));
            }
        }
        Some(format!("tool '{tool}' is not on the pipe agent allowlist"))
    }

    /// `/mcp <name> allow <tool>`. Nested Pipe chat/responses stay denied.
    pub fn allow_tool(
        &mut self,
        source: McpConfigSource,
        server: &str,
        tool: &str,
    ) -> Result<(), String> {
        if tool.trim().is_empty() {
            return Err("tool name is empty".into());
        }
        if source == McpConfigSource::Pipe && is_never_agent_pipe_tool(tool) {
            return Err(format!(
                "tool '{tool}' is not agent-callable on server '{server}' \
                 (nested model calls stay off the coding loop)"
            ));
        }
        let entry = self.servers.entry(server.to_string()).or_default();
        entry.exclude.retain(|denied| denied != tool);
        if source == McpConfigSource::Pipe {
            if !PIPE_DEFAULT_ALLOWED_TOOLS.contains(&tool)
                && !self.extra_allow.iter().any(|allowed| allowed == tool)
            {
                self.extra_allow.push(tool.to_string());
            }
        } else if !entry.only.is_empty() && !entry.only.iter().any(|allowed| allowed == tool) {
            entry.only.push(tool.to_string());
        }
        Ok(())
    }

    /// `/mcp <name> deny <tool>`.
    pub fn deny_tool(
        &mut self,
        source: McpConfigSource,
        server: &str,
        tool: &str,
    ) -> Result<(), String> {
        if tool.trim().is_empty() {
            return Err("tool name is empty".into());
        }
        if source == McpConfigSource::Pipe && is_never_agent_pipe_tool(tool) {
            return Ok(());
        }
        let entry = self.servers.entry(server.to_string()).or_default();
        entry.only.retain(|allowed| allowed != tool);
        if !entry.exclude.iter().any(|denied| denied == tool) {
            entry.exclude.push(tool.to_string());
        }
        if source == McpConfigSource::Pipe {
            self.extra_allow.retain(|allowed| allowed != tool);
        }
        Ok(())
    }
}

pub fn is_never_agent_pipe_tool(name: &str) -> bool {
    PIPE_NEVER_AGENT_TOOLS.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_servers_are_unrestricted() {
        let policy = AgentToolPolicy::new();
        assert!(policy.allows(McpConfigSource::Hi, "docs", "anything"));
        assert!(policy.allows(
            McpConfigSource::Claude,
            "imported",
            "pipe.chat.completions.create"
        ));
    }

    #[test]
    fn imported_only_and_exclude() {
        let mut policy = AgentToolPolicy::new();
        policy.set_from_config("docs", &["search".into()], &["delete".into()]);
        assert!(policy.allows(McpConfigSource::Hi, "docs", "search"));
        assert!(!policy.allows(McpConfigSource::Hi, "docs", "other"));
        assert!(!policy.allows(McpConfigSource::Hi, "docs", "delete"));
        policy.overlay(
            "docs",
            &ServerAllowList {
                only: Vec::new(),
                exclude: vec!["search".into()],
            },
        );
        assert!(!policy.allows(McpConfigSource::Hi, "docs", "search"));
    }

    #[test]
    fn pipe_defaults_allow_list_and_health_only() {
        let policy = AgentToolPolicy::new();
        assert!(policy.allows(McpConfigSource::Pipe, PIPE_SERVER_NAME, "pipe.models.list"));
        assert!(policy.allows(
            McpConfigSource::Pipe,
            PIPE_SERVER_NAME,
            "pipe.models.health"
        ));
        assert!(!policy.allows(
            McpConfigSource::Pipe,
            PIPE_SERVER_NAME,
            "pipe.usage.summary"
        ));
        assert!(!policy.allows(McpConfigSource::Pipe, PIPE_SERVER_NAME, "pipe.request.get"));
        assert!(!policy.allows(
            McpConfigSource::Pipe,
            PIPE_SERVER_NAME,
            "pipe.chat.completions.create"
        ));
        assert!(!policy.allows(
            McpConfigSource::Pipe,
            PIPE_SERVER_NAME,
            "pipe.responses.create"
        ));
    }

    #[test]
    fn extra_allow_is_additive_but_cannot_enable_nested_chat() {
        let policy = AgentToolPolicy::with_pipe_extra_allow([
            "pipe.usage.summary",
            "pipe.chat.completions.create",
            "pipe.responses.create",
        ]);
        assert!(policy.allows(McpConfigSource::Pipe, PIPE_SERVER_NAME, "pipe.models.list"));
        assert!(policy.allows(
            McpConfigSource::Pipe,
            PIPE_SERVER_NAME,
            "pipe.usage.summary"
        ));
        assert!(!policy.allows(
            McpConfigSource::Pipe,
            PIPE_SERVER_NAME,
            "pipe.chat.completions.create"
        ));
        assert!(!policy.allows(
            McpConfigSource::Pipe,
            PIPE_SERVER_NAME,
            "pipe.responses.create"
        ));
        assert!(
            policy
                .deny_reason(
                    McpConfigSource::Pipe,
                    PIPE_SERVER_NAME,
                    "pipe.chat.completions.create"
                )
                .unwrap()
                .contains("nested model")
        );
        assert!(
            policy
                .deny_reason(McpConfigSource::Pipe, PIPE_SERVER_NAME, "pipe.request.get")
                .unwrap()
                .contains("allowlist")
        );
    }

    #[test]
    fn slash_allow_deny_imported() {
        let mut policy = AgentToolPolicy::new();
        policy
            .deny_tool(McpConfigSource::Hi, "docs", "wipe")
            .unwrap();
        assert!(!policy.allows(McpConfigSource::Hi, "docs", "wipe"));
        policy
            .allow_tool(McpConfigSource::Hi, "docs", "wipe")
            .unwrap();
        assert!(policy.allows(McpConfigSource::Hi, "docs", "wipe"));
        policy.set_from_config("docs", &["search".into()], &[]);
        assert!(!policy.allows(McpConfigSource::Hi, "docs", "other"));
        policy
            .allow_tool(McpConfigSource::Hi, "docs", "other")
            .unwrap();
        assert!(policy.allows(McpConfigSource::Hi, "docs", "other"));
    }
}
