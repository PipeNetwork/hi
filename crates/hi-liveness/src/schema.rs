//! Shared heartbeat, event, and turn-intent records.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

pub const ENV_SUPERVISED: &str = "HI_SENTINEL_SUPERVISED";
pub const ENV_GENERATION: &str = "HI_SENTINEL_GENERATION";
pub const ENV_ROLE: &str = "HI_SENTINEL_ROLE";
pub const ENV_INSTANCE: &str = "HI_SENTINEL_INSTANCE";
pub const ENV_HEARTBEAT: &str = "HI_SENTINEL_HEARTBEAT";
pub const ENV_EVENTS: &str = "HI_SENTINEL_EVENTS";
pub const ENV_TURN_INTENT: &str = "HI_SENTINEL_TURN_INTENT";
pub const ENV_HI_BINARY: &str = "HI_SENTINEL_HI_BINARY";
pub const ENV_RESUME_INCOMPLETE: &str = "HI_SENTINEL_RESUME_INCOMPLETE";
pub const ENV_PANIC_FILE: &str = "HI_SENTINEL_PANIC_FILE";
pub const ENV_CRASH_DIR: &str = "HI_CRASH_DIR";

pub const HEARTBEAT_PERIOD_MS: u64 = 500;
pub const EVENT_LOG_CAP: u64 = 2 * 1024 * 1024;

pub const IDENTICAL_TOOL_CONSECUTIVE: u32 = 8;
pub const IDENTICAL_TOOL_IN_TURN: u32 = 17;
pub const IDENTICAL_TOOL_ERROR_REPEAT: u32 = 9;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HarnessState {
    #[default]
    Starting,
    AwaitingUser,
    Idle,
    AwaitingModel,
    ExecutingTool,
    AwaitingConfirmation,
    Compacting,
    Verifying,
    ShuttingDown,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InvariantCode {
    ToolUnclosed,
    TurnUnclosed,
    SessionAppendFailed,
    ChildLeak,
    ConfirmUnanswered,
    IdenticalToolStorm,
    /// Model stopped after executing tools with no user-visible text. The
    /// harness continues a bounded number of times, then fails the turn.
    /// This is model behavior, not a crash: report-only, never auto-repair.
    /// Killing an idle TUI over an empty stop is worse than leaving `/retry`.
    EmptyAssistantAfterTools,
    /// Model compact failed while occupancy was already over the window.
    /// Identical-tool storms in that state are a symptom: the harness must
    /// auto-repair rather than sit in ReportOnly forever.
    CompactFailedOverWindow,
}

impl InvariantCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolUnclosed => "tool_unclosed",
            Self::TurnUnclosed => "turn_unclosed",
            Self::SessionAppendFailed => "session_append_failed",
            Self::ChildLeak => "child_leak",
            Self::ConfirmUnanswered => "confirm_unanswered",
            Self::IdenticalToolStorm => "identical_tool_storm",
            Self::EmptyAssistantAfterTools => "empty_assistant_after_tools",
            Self::CompactFailedOverWindow => "compact_failed_over_window",
        }
    }
}

/// Codes the supervisor may treat as a harness bug. A plain
/// `IdenticalToolStorm` (agent loop under a healthy window) is excluded;
/// the same storm after compact has already failed is `CompactFailedOverWindow`.
/// `EmptyAssistantAfterTools` is also excluded: the turn already ended with
/// an error the user can `/retry`. SIGTERM of that idle TUI is worse.
pub const AUTO_REPAIR_SET: &[InvariantCode] = &[
    InvariantCode::ToolUnclosed,
    InvariantCode::TurnUnclosed,
    InvariantCode::SessionAppendFailed,
    InvariantCode::ChildLeak,
    InvariantCode::ConfirmUnanswered,
    InvariantCode::CompactFailedOverWindow,
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InvariantViolation {
    pub code: InvariantCode,
    pub ts_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventCode {
    ToolStart,
    ToolEnd,
    TurnStart,
    TurnEnd,
    TurnError,
    TurnCancel,
    State,
    CompactStart,
    CompactEnd,
    VerifyStart,
    VerifyEnd,
    ConfirmShown,
    ConfirmAnswered,
    SessionAppend,
    SessionRewrite,
    Invariant,
    ChildSpawn,
    ChildWait,
    Progress,
    ShuttingDown,
}

impl EventCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolStart => "tool_start",
            Self::ToolEnd => "tool_end",
            Self::TurnStart => "turn_start",
            Self::TurnEnd => "turn_end",
            Self::TurnError => "turn_error",
            Self::TurnCancel => "turn_cancel",
            Self::State => "state",
            Self::CompactStart => "compact_start",
            Self::CompactEnd => "compact_end",
            Self::VerifyStart => "verify_start",
            Self::VerifyEnd => "verify_end",
            Self::ConfirmShown => "confirm_shown",
            Self::ConfirmAnswered => "confirm_answered",
            Self::SessionAppend => "session_append",
            Self::SessionRewrite => "session_rewrite",
            Self::Invariant => "invariant",
            Self::ChildSpawn => "child_spawn",
            Self::ChildWait => "child_wait",
            Self::Progress => "progress",
            Self::ShuttingDown => "shutting_down",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LiveEvent {
    pub ts_unix_ms: u64,
    pub code: EventCode,
    pub state: HarnessState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Heartbeat {
    pub schema_version: u32,
    pub seq: u64,
    pub ts_unix_ms: u64,
    pub pid: u32,
    pub instance: String,
    pub generation: u32,
    pub state: HarnessState,
    pub last_progress_unix_ms: u64,
    pub last_event: Option<String>,
    pub last_tool: Option<String>,
    pub last_tool_id: Option<String>,
    pub consecutive_identical_tools: u32,
    pub identical_tool_count_in_turn: u32,
    pub turn_index: u32,
    pub session_path: Option<String>,
    pub workspace: String,
    pub pre_checkpoint: Option<String>,
    pub child_pgids: Vec<i32>,
    pub current_tool_pgid: Option<i32>,
    pub invariant: Option<InvariantViolation>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TurnIntent {
    pub schema_version: u32,
    pub turn_index: u32,
    pub prompt: String,
    pub session_path: Option<String>,
    pub pre_checkpoint: Option<String>,
    pub started_unix_ms: u64,
    pub workspace: String,
    pub oneshot: bool,
    pub plain: bool,
}

pub fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn env_flag_on(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Publish only from the process that started the writer, and only while the
/// supervisor-minted instance token still matches (nested `hi` must not clobber
/// the parent's file).
pub fn writer_may_publish(pid: u32, instance: &str) -> bool {
    if pid != std::process::id() {
        return false;
    }
    match std::env::var(ENV_INSTANCE) {
        Ok(env_instance) => env_instance == instance,
        Err(_) => true,
    }
}
