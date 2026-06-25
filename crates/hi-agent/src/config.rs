//! Per-session agent configuration and the layered-verification stage type.

use hi_ai::{CompatMode, ToolMode};

use crate::compaction::{CompactionKind, DEFAULT_KEEP_RECENT};
use crate::{
    AUTO_COMPACT_PERCENT, COMPACT_TARGET_PERCENT, IN_TURN_ELIDE_PERCENT,
    IN_TURN_KEEP_TOOL_RESULTS, MAX_CONTINUE_NUDGES, MAX_EMPTY_RETRIES, MAX_PARALLEL_TOOLS,
    MAX_REPEAT_NUDGES,
};

/// One stage of layered verification: a short label and the shell command to
/// run. Stages run in order; the first to fail stops the turn and its output is
/// fed back to the model. A cheap compile/typecheck stage before tests yields
/// fast, localizable errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyStage {
    pub name: String,
    pub command: String,
}

impl VerifyStage {
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
        }
    }

    /// Whether this stage runs tests (vs. a compile/lint/typecheck gate) — used
    /// to tailor the failure guidance fed back to the model.
    pub(crate) fn is_test(&self) -> bool {
        let n = self.name.to_lowercase();
        n.contains("test") || n.contains("spec")
    }
}

/// Per-session configuration the agent applies to every request.
pub struct AgentConfig {
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub thinking_budget: Option<u32>,
    pub tool_mode: ToolMode,
    pub compat: CompatMode,
    /// USD per 1M (input, output) tokens, when known — used for cost display.
    pub price: Option<(f64, f64)>,
    /// Model context window, when known — used to show how full it is.
    pub context_window: Option<u32>,
    /// Project context (e.g. from HI.md/AGENTS.md) appended to the system prompt.
    pub project_context: Option<String>,
    /// Ordered verification stages run after the model stops — a cheap
    /// compile/typecheck first, then lint, then tests. The first stage to fail
    /// stops the turn and its output is fed back so the model iterates
    /// (verification-in-the-loop). Empty = verification off.
    pub verify: Vec<VerifyStage>,
    /// Cap on verification retry rounds.
    pub max_verify_iterations: u32,
    /// Safety cap on model calls per turn, to stop runaway tool loops.
    pub max_steps: u32,
    /// When the context window fills past a threshold, summarize-and-reset
    /// before the next turn so a long session doesn't overflow the model.
    pub auto_compact: bool,
    /// Strategy used by `/compact` (no arg) and the summarizing tier of
    /// auto-compaction.
    pub compaction: CompactionKind,
    /// After a turn that changed files, make one dedicated tool-free model call
    /// to produce a structured recap — so even a model that won't summarize on
    /// its own ends with one. Costs one extra call per file-changing turn.
    pub finalize: bool,
    /// Max times one turn will nudge a model that stopped after *describing* a
    /// next step without performing it. Default: [`MAX_CONTINUE_NUDGES`].
    pub max_continue_nudges: u32,
    /// Max times one turn will nudge a model that re-issues the exact same tool
    /// call as the previous round (a repetition loop). Default:
    /// [`MAX_REPEAT_NUDGES`].
    pub max_repeat_nudges: u32,
    /// How many times to silently re-run a round that produced no usable output.
    /// Default: [`MAX_EMPTY_RETRIES`].
    pub max_empty_retries: u32,
    /// Max read-only tool calls to run concurrently within one round.
    /// Default: [`MAX_PARALLEL_TOOLS`].
    pub max_parallel_tools: usize,
    /// Auto-compact once the context window is at least this percent full.
    /// Default: [`AUTO_COMPACT_PERCENT`].
    pub auto_compact_percent: u64,
    /// After triggering, compact until the local estimate is at or below this
    /// percent of the window. Default: [`COMPACT_TARGET_PERCENT`].
    pub compact_target_percent: u64,
    /// During one long tool loop, begin dropping old bulky tool payloads before
    /// the next model call. Default: [`IN_TURN_ELIDE_PERCENT`].
    pub in_turn_elide_percent: u64,
    /// Keep the newest tool results verbatim when trimming inside a turn.
    /// Default: [`IN_TURN_KEEP_TOOL_RESULTS`].
    pub in_turn_keep_tool_results: usize,
    /// Whether to run a per-file fast check (syntax/lint) in the background
    /// right after a write/edit, so errors surface during the turn instead of
    /// only at turn-end verify. Off by default; only fires for languages with a
    /// genuinely per-file fast check (see `hi_tools::fast_check_for`).
    pub proactive_verify: bool,
    /// Whether long-horizon agency is on: a structured `Goal` the agent
    /// decomposes into sub-goals, drives across turns, retries on failure, and
    /// resumes across sessions. Off by default while it stabilizes; when off,
    /// the agent behaves as the single-turn loop.
    pub long_horizon: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_tokens: 4096,
            temperature: None,
            thinking_budget: None,
            tool_mode: ToolMode::Auto,
            compat: CompatMode::Auto,
            price: None,
            context_window: None,
            project_context: None,
            verify: Vec::new(),
            max_verify_iterations: 2,
            max_steps: 500,
            auto_compact: true,
            compaction: CompactionKind::ElideThenSummarizeTail {
                keep_recent: DEFAULT_KEEP_RECENT,
            },
            finalize: true,
            max_continue_nudges: MAX_CONTINUE_NUDGES,
            max_repeat_nudges: MAX_REPEAT_NUDGES,
            max_empty_retries: MAX_EMPTY_RETRIES,
            max_parallel_tools: MAX_PARALLEL_TOOLS,
            auto_compact_percent: AUTO_COMPACT_PERCENT,
            compact_target_percent: COMPACT_TARGET_PERCENT,
            in_turn_elide_percent: IN_TURN_ELIDE_PERCENT,
            in_turn_keep_tool_results: IN_TURN_KEEP_TOOL_RESULTS,
            proactive_verify: false,
            long_horizon: false,
        }
    }
}
