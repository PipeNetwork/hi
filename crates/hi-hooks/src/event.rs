//! Hook event types and payloads.

use serde::{Deserialize, Serialize};

/// Hook event types. Each maps to a point in the agent lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEvent {
    SessionStart,
    SessionEnd,
    /// Fires before a tool is executed. Can block (deny) the tool call.
    PreToolUse,
    /// Fires after a tool completes. Observe-only.
    PostToolUse,
    /// Fires when the user submits a prompt. Observe-only.
    UserPromptSubmit,
    /// Fires before a subagent (delegate/explore) starts.
    SubagentStart,
    /// Fires after a subagent completes.
    SubagentStop,
}

/// The payload sent to a hook command via stdin (JSON).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HookEventEnvelope {
    pub hook_event: String,
    pub session_id: String,
    pub workspace_root: String,
    pub timestamp: String,
    pub payload: HookPayload,
}

/// Event-specific payload data.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum HookPayload {
    SessionStart {
        source: String,
    },
    SessionEnd,
    PreToolUse {
        tool_name: String,
        arguments: serde_json::Value,
    },
    PostToolUse {
        tool_name: String,
        arguments: serde_json::Value,
        status: String,
        output: String,
    },
    UserPromptSubmit {
        prompt: String,
    },
    SubagentStart {
        task: String,
    },
    SubagentStop {
        task: String,
        status: String,
    },
}

impl HookEvent {
    /// Whether this event can block agent actions (only `PreToolUse` is blocking).
    pub fn is_blocking(self) -> bool {
        matches!(self, HookEvent::PreToolUse)
    }
}
