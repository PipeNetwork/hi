//! Hook result types.

use std::time::Duration;

/// The outcome of a blocking (`pre_tool_use`) hook dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookDecision {
    Allow,
    Deny {
        reason: String,
        hook_name: String,
    },
}

/// The outcome of a single hook execution.
#[derive(Debug, Clone)]
pub enum HookRunResult {
    Success {
        hook_name: String,
        elapsed: Duration,
    },
    Skipped {
        hook_name: String,
    },
    /// Hook failed (timeout, crash, bad output): fail-open.
    Failed {
        hook_name: String,
        error: String,
        elapsed: Duration,
    },
}
