//! Per-session agent configuration and the layered-verification stage type.

use hi_ai::{CompatMode, ReasoningEffort, ToolMode};
use serde::{Deserialize, Serialize};

use crate::compaction::{CompactionKind, DEFAULT_KEEP_RECENT};
use crate::{
    AUTO_COMPACT_PERCENT, COMPACT_TARGET_PERCENT, IN_TURN_ELIDE_PERCENT, IN_TURN_KEEP_TOOL_RESULTS,
    MAX_EMPTY_RETRIES, MAX_PARALLEL_TOOLS, MAX_REPEAT_NUDGES, MAX_SILENT_CONTINUES,
    MAX_TRUNCATION_RETRIES,
};

/// One stage of layered verification: a short label and the shell command to
/// run. Stages run in order; the first to fail stops the turn and its output is
/// fed back to the model. A cheap compile/typecheck stage before tests yields
/// fast, localizable errors.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifyStage {
    pub name: String,
    pub command: String,
}

/// How deterministic verification is selected for a turn.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "stages", rename_all = "snake_case")]
pub enum VerificationMode {
    /// Detect a project-appropriate pipeline from the workspace.
    #[default]
    Auto,
    /// Run exactly these stages, in order.
    Explicit(Vec<VerifyStage>),
    /// Do not run deterministic verification. Mutations remain unverified.
    Disabled,
}

impl VerificationMode {
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Self::Explicit(stages) = self {
            anyhow::ensure!(
                !stages.is_empty() && stages.iter().all(|stage| !stage.command.trim().is_empty()),
                "explicit verification requires non-empty command stages"
            );
        }
        Ok(())
    }

    pub fn resolved_stages(&self, root: &std::path::Path) -> Vec<VerifyStage> {
        match self {
            Self::Auto => detect_verify_pipeline(root),
            Self::Explicit(stages) => stages.clone(),
            Self::Disabled => Vec::new(),
        }
    }
}

/// Independent-review policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewPolicy {
    #[default]
    Risk,
    Always,
    Off,
}

/// Workspace-local language-server policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LspMode {
    #[default]
    Auto,
    On,
    Off,
}

/// When the write-capable `delegate` subagent is advertised.
///
/// Depth is always capped at 1 (children never get `delegate`). This policy only
/// controls the *parent* advertisement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteSubagentPolicy {
    /// Never advertise `delegate` (explicit `/delegate off` or profile false).
    Off,
    /// Advertise only for multi-file / isolation-shaped mutation tasks (default).
    /// Small single-file fixes stay in-process; risky handoffs get worktree isolation.
    #[default]
    Risk,
    /// Advertise on every mutation-capable turn (`/delegate on`, `HI_WRITE_SUBAGENTS`).
    On,
}

impl WriteSubagentPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Risk => "risk",
            Self::On => "on",
        }
    }

    /// True when the tool may be injected for some tasks (not hard-off).
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// Tool advertisement policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSet {
    #[default]
    Dynamic,
    Minimal,
    Full,
}

impl ReviewPolicy {
    pub fn label(self) -> &'static str {
        match self {
            ReviewPolicy::Risk => "risk",
            ReviewPolicy::Always => "always",
            ReviewPolicy::Off => "off",
        }
    }
}

impl LspMode {
    pub fn label(self) -> &'static str {
        match self {
            LspMode::Auto => "auto",
            LspMode::On => "on",
            LspMode::Off => "off",
        }
    }
}

impl ToolSet {
    pub fn label(self) -> &'static str {
        match self {
            ToolSet::Dynamic => "dynamic",
            ToolSet::Minimal => "minimal",
            ToolSet::Full => "full",
        }
    }
}

/// Guess a layered deterministic verification pipeline from marker files.
pub fn detect_verify_pipeline(dir: &std::path::Path) -> Vec<VerifyStage> {
    let has = |name: &str| dir.join(name).exists();
    let stage = |name: &str, command: &str| VerifyStage::new(name, command);
    if has("Cargo.toml") {
        vec![
            stage("check", "cargo check --quiet"),
            stage("test", "cargo test --quiet"),
        ]
    } else if has("go.mod") {
        vec![
            stage("build", "go build ./..."),
            stage("test", "go test ./..."),
        ]
    } else if has("package.json") {
        let mut stages = Vec::new();
        if has("tsconfig.json") {
            stages.push(stage("typecheck", "npx --no-install tsc --noEmit"));
        }
        stages.push(stage("test", "npm test --silent"));
        stages
    } else if has("pyproject.toml") || has("setup.py") || has("pytest.ini") || has("tox.ini") {
        let mut stages = Vec::new();
        if has("ruff.toml") || has(".ruff.toml") {
            stages.push(stage("lint", "ruff check ."));
        }
        stages.push(stage("test", "pytest -q"));
        stages
    } else if has("Makefile") || has("makefile") {
        vec![stage("test", "make test")]
    } else {
        Vec::new()
    }
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
/// `Clone` so a fleet/dashboard can stamp out additional agents from the
/// session's resolved config (tweaking per-agent fields as needed).
///
/// Fields are grouped by concern so related knobs stay together:
/// `paths`, `routing`, `gates`, `loop_limits`, `memory`, `subagents`, `rsi`.
#[derive(Clone)]
pub struct AgentConfig {
    /// Workspace and state roots.
    pub paths: AgentPaths,
    /// Model route, sampling, tool mode, context window.
    pub routing: AgentRouting,
    /// Verification, review, LSP, and mutation safety gates.
    pub gates: AgentGates,
    /// Per-turn step / retry / parallelism caps.
    pub loop_limits: AgentLoopLimits,
    /// Compaction, finalize, project context, tool-set selection.
    pub memory: AgentMemory,
    /// Explore/delegate/planner/skeptic subagent policy.
    pub subagents: AgentSubagents,
    /// Optional RSI control-plane hooks (interactive path stays thin).
    pub rsi: AgentRsi,
}

/// Explicit workspace and durable-state roots.
#[derive(Clone, Debug)]
pub struct AgentPaths {
    /// Explicit workspace root for tools, verification, LSP, and checkpoints.
    pub workspace_root: std::path::PathBuf,
    /// Per-workspace internal snapshots, journals, and indexes.
    pub state_root: std::path::PathBuf,
}

impl Default for AgentPaths {
    fn default() -> Self {
        Self {
            workspace_root: std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            state_root: std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join(".hi"),
        }
    }
}

/// Model identity, sampling, and provider routing.
#[derive(Clone, Debug)]
pub struct AgentRouting {
    pub model: String,
    /// Human-readable effective provider route, when known by the frontend.
    pub provider_route: Option<String>,
    /// The user/config requested output-token cap before live model metadata is
    /// applied. Kept separately so `/model` switches can recompute the active
    /// cap without inheriting the previous route's live limit.
    pub requested_max_tokens: u32,
    pub max_tokens: u32,
    /// True when the user deliberately set the cap (CLI or non-default profile).
    /// Explicit caps are honored, only clamped downward to a model's advertised
    /// limit.
    pub max_tokens_explicit: bool,
    pub temperature: Option<f32>,
    pub thinking_budget: Option<u32>,
    /// Abstract reasoning level (`reasoning_effort`) applied to every main-turn
    /// request on OpenAI-compatible endpoints that support it; `None` leaves the
    /// endpoint default. See [`hi_ai::ReasoningEffort`]. Housekeeping calls
    /// (compaction/memory/recap) deliberately leave this off. Set via
    /// `--reasoning-effort`, a profile, or `/config reasoning <level>`.
    pub reasoning_effort: Option<ReasoningEffort>,
    pub tool_mode: ToolMode,
    pub compat: CompatMode,
    /// Model context window, when known — used to show how full it is.
    pub context_window: Option<u32>,
}

impl Default for AgentRouting {
    fn default() -> Self {
        Self {
            model: String::new(),
            provider_route: None,
            requested_max_tokens: 8192,
            max_tokens: 8192,
            max_tokens_explicit: false,
            temperature: None,
            thinking_budget: None,
            reasoning_effort: None,
            tool_mode: ToolMode::default(),
            compat: CompatMode::default(),
            context_window: None,
        }
    }
}

/// Quality and safety gates around mutations and answers.
#[derive(Clone, Debug)]
pub struct AgentGates {
    /// Automatic, explicit, or disabled deterministic verification.
    pub verification: VerificationMode,
    /// Repair/check cycles allowed after the initial verification check.
    pub max_verify_repairs: u32,
    /// Independent-review policy.
    pub review: ReviewPolicy,
    /// Permit a mutation turn to complete with `Unverified` status.
    pub allow_unverified: bool,
    /// Permit mutation when no Git or internal checkpoint backend is available.
    pub allow_no_checkpoint: bool,
    /// Whether to run a per-file fast check (syntax/lint) in the background
    /// right after a write/edit, so errors surface during the turn instead of
    /// only at turn-end verify. Off by default; only fires for languages with a
    /// genuinely per-file fast check (see `hi_tools::fast_check_for`).
    ///
    /// Independent of the always-on mid-turn Rust path (LSP diagnostics +
    /// affected-package `cargo check` after batches that mutate `.rs` files).
    pub proactive_verify: bool,
    /// Whether read-only review/status/security/gap turns get a deterministic
    /// inspection seed before the first model call.
    pub read_only_preflight: bool,
    /// When true, ask the user to confirm each write/edit before applying.
    pub confirm_edits: bool,
    /// Workspace-local language-server policy.
    pub lsp_mode: LspMode,
}

impl Default for AgentGates {
    fn default() -> Self {
        Self {
            verification: VerificationMode::Auto,
            max_verify_repairs: 2,
            review: ReviewPolicy::Risk,
            allow_unverified: false,
            allow_no_checkpoint: true,
            proactive_verify: false,
            read_only_preflight: true,
            confirm_edits: false,
            lsp_mode: LspMode::Auto,
        }
    }
}

/// Caps that bound a single turn's model/tool loops.
#[derive(Clone, Debug)]
pub struct AgentLoopLimits {
    /// Safety cap on model calls per turn, to stop runaway tool loops.
    pub max_steps: u32,
    /// Whether `max_steps` was explicitly requested by the caller. When false,
    /// the turn loop chooses a conservative dynamic cap from the turn intent.
    pub max_steps_explicit: bool,
    /// Hard cap on executed tool calls per turn. This is independent of the
    /// model-call (`max_steps`) cap.
    pub max_tool_calls: u32,
    /// Max times one turn will nudge a model that re-issues the exact same tool
    /// call as the previous round (a repetition loop). Default:
    /// [`MAX_REPEAT_NUDGES`].
    pub max_repeat_nudges: u32,
    /// Max times a turn will silently re-prompt the model to continue after it
    /// stops with text but no tool calls. Default: [`MAX_SILENT_CONTINUES`].
    pub max_silent_continues: u32,
    /// How many times to silently re-run a round that produced no usable output.
    /// Default: [`MAX_EMPTY_RETRIES`].
    pub max_empty_retries: u32,
    /// Max times one turn will nudge the model to continue after its output was
    /// truncated by the output token cap. Default: [`MAX_TRUNCATION_RETRIES`].
    pub max_truncation_retries: u32,
    /// Max read-only tool calls to run concurrently within one round.
    /// Default: [`MAX_PARALLEL_TOOLS`].
    pub max_parallel_tools: usize,
    /// Per-mode budgets for **review-answer** repair during Steer (not workspace
    /// compile/lint/test repair — that is [`AgentGates::max_verify_repairs`]).
    pub review_repair: ReviewRepairBudgets,
}

impl Default for AgentLoopLimits {
    fn default() -> Self {
        Self {
            max_steps: u32::MAX,
            max_steps_explicit: false,
            max_tool_calls: u32::MAX,
            max_repeat_nudges: MAX_REPEAT_NUDGES,
            max_silent_continues: MAX_SILENT_CONTINUES,
            max_empty_retries: MAX_EMPTY_RETRIES,
            max_truncation_retries: MAX_TRUNCATION_RETRIES,
            max_parallel_tools: MAX_PARALLEL_TOOLS,
            review_repair: ReviewRepairBudgets::default(),
        }
    }
}

/// How many times each review-answer repair mode may fire in one turn.
///
/// Defaults match the historical hard-coded mode limits. Operators can lower
/// them for cheaper/stricter sessions or raise them for stubborn models.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReviewRepairBudgets {
    pub no_evidence: u32,
    pub listing_only: u32,
    pub generic_template: u32,
    pub inspected_disclaimer: u32,
    pub inspected_disclaimer_chat_attempt: u32,
    pub concrete_answer: u32,
    pub read_after_search: u32,
    pub security_broad_search: u32,
    pub security_scope: u32,
    pub gap_search_overclaim: u32,
}

impl Default for ReviewRepairBudgets {
    fn default() -> Self {
        Self {
            no_evidence: 4,
            listing_only: 4,
            generic_template: 4,
            inspected_disclaimer: 4,
            inspected_disclaimer_chat_attempt: 2,
            concrete_answer: 4,
            read_after_search: 2,
            security_broad_search: 4,
            security_scope: 5,
            gap_search_overclaim: 3,
        }
    }
}

impl ReviewRepairBudgets {
    /// Budget for a stable review-repair mode key (`review_no_evidence`, …).
    pub fn limit_for_key(&self, key: &str) -> u32 {
        match key {
            "review_no_evidence" => self.no_evidence,
            "review_listing_only" => self.listing_only,
            "review_generic_template" => self.generic_template,
            "review_inspected_disclaimer" => self.inspected_disclaimer,
            "review_inspected_disclaimer_chat_attempt" => self.inspected_disclaimer_chat_attempt,
            "review_concrete_answer" => self.concrete_answer,
            "review_read_after_search" => self.read_after_search,
            "review_security_broad_search" => self.security_broad_search,
            "review_security_scope" => self.security_scope,
            "review_gap_search_overclaim" => self.gap_search_overclaim,
            // Unknown keys get a conservative default rather than unlimited.
            _ => 2,
        }
    }
}

/// Context window management, project context, and tool catalog selection.
#[derive(Clone, Debug)]
pub struct AgentMemory {
    /// Project context (e.g. from HI.md/AGENTS.md) appended to the system prompt.
    pub project_context: Option<String>,
    /// When the context window fills past a threshold, summarize-and-reset
    /// before the next turn so a long session doesn't overflow the model.
    pub auto_compact: bool,
    /// Strategy used by `/compact` (no arg) and the summarizing tier of
    /// auto-compaction.
    pub compaction: CompactionKind,
    /// After a turn that changed files, make one dedicated tool-free model call
    /// to produce a structured recap.
    pub finalize: bool,
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
    /// Which built-in tools are advertised to the model.
    pub tool_set: ToolSet,
    /// Glob-style path exclusions applied when ranking repository context.
    pub context_exclusions: Vec<String>,
    /// Whether the agent may curate/learn skills during the session.
    pub curate_skills: bool,
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self {
            project_context: None,
            auto_compact: true,
            compaction: CompactionKind::ElideThenSummarizeTail {
                keep_recent: DEFAULT_KEEP_RECENT,
            },
            finalize: true,
            auto_compact_percent: AUTO_COMPACT_PERCENT,
            compact_target_percent: COMPACT_TARGET_PERCENT,
            in_turn_elide_percent: IN_TURN_ELIDE_PERCENT,
            in_turn_keep_tool_results: IN_TURN_KEEP_TOOL_RESULTS,
            tool_set: ToolSet::Dynamic,
            context_exclusions: Vec::new(),
            curate_skills: false,
        }
    }
}

/// Subagent and multi-model planning policy.
#[derive(Clone, Debug)]
pub struct AgentSubagents {
    /// Read-only, depth-capped explore children (safe default-on for coding).
    pub explore_subagents: bool,
    /// When the write-capable `delegate` subagent is advertised.
    pub write_subagents: WriteSubagentPolicy,
    /// True when this agent instance is itself a subagent child.
    pub is_subagent: bool,
    /// Whether long-horizon agency is on: a structured `Goal` the agent
    /// decomposes into sub-goals across turns.
    pub long_horizon: bool,
    /// Model id used to decompose a `/goal <objective>` into sub-goals.
    pub planner_model: Option<String>,
    /// Model id used by the `/goal team` skeptic gate.
    pub skeptic_model: Option<String>,
    /// Optional OpenAI-compatible base URL for the skeptic review call only.
    pub skeptic_endpoint: Option<String>,
    /// API key sent to `skeptic_endpoint`.
    pub skeptic_endpoint_key: Option<String>,
}

impl Default for AgentSubagents {
    fn default() -> Self {
        Self {
            explore_subagents: true,
            write_subagents: WriteSubagentPolicy::Risk,
            is_subagent: false,
            long_horizon: false,
            planner_model: None,
            skeptic_model: None,
            skeptic_endpoint: None,
            skeptic_endpoint_key: None,
        }
    }
}

/// Optional RSI hooks supplied by the frontend (not the interactive turn SM).
#[derive(Clone, Default)]
pub struct AgentRsi {
    /// Candidate-side evidence requested for subsequent turns.
    pub enabled: bool,
    /// Managed mode is immutable from the interactive configuration surface.
    pub managed: bool,
    /// Shared remote-provider switch. Absent in managed workers and when no Pipe
    /// credentials were available at startup.
    pub remote_switch: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Public RSI recovery and capability operations supplied by the frontend.
    pub control: Option<std::sync::Arc<dyn crate::RsiControl>>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            paths: AgentPaths::default(),
            routing: AgentRouting::default(),
            gates: AgentGates::default(),
            loop_limits: AgentLoopLimits::default(),
            memory: AgentMemory::default(),
            subagents: AgentSubagents::default(),
            rsi: AgentRsi::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_defaults_are_safe_and_automatic() {
        let config = AgentConfig::default();
        assert_eq!(config.gates.verification, VerificationMode::Auto);
        assert_eq!(config.gates.max_verify_repairs, 2);
        assert_eq!(config.gates.review, ReviewPolicy::Risk);
        assert_eq!(config.gates.lsp_mode, LspMode::Auto);
        assert_eq!(config.memory.tool_set, ToolSet::Dynamic);
        assert!(!config.loop_limits.max_steps_explicit);
        assert!(!config.gates.allow_unverified);
        assert!(config.gates.allow_no_checkpoint);
        assert!(config.subagents.explore_subagents, "explore on by default");
        assert_eq!(config.subagents.write_subagents, WriteSubagentPolicy::Risk);
        let budgets = &config.loop_limits.review_repair;
        assert_eq!(budgets.no_evidence, 4);
        assert_eq!(budgets.read_after_search, 2);
        assert_eq!(budgets.security_scope, 5);
        assert_eq!(budgets.gap_search_overclaim, 3);
        assert_eq!(budgets.limit_for_key("review_listing_only"), 4);
    }
}
