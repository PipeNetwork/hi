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

/// Tool advertisement policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSet {
    #[default]
    Dynamic,
    Minimal,
    Full,
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
#[derive(Clone)]
pub struct AgentConfig {
    /// Explicit workspace root for tools, verification, LSP, and checkpoints.
    pub workspace_root: std::path::PathBuf,
    /// Per-workspace internal snapshots, journals, and indexes.
    pub state_root: std::path::PathBuf,
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
    /// Project context (e.g. from HI.md/AGENTS.md) appended to the system prompt.
    pub project_context: Option<String>,
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
    /// Safety cap on model calls per turn, to stop runaway tool loops.
    pub max_steps: u32,
    /// Whether `max_steps` was explicitly requested by the caller. When false,
    /// the turn loop chooses a conservative dynamic cap from the turn intent.
    pub max_steps_explicit: bool,
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
    /// Max times one turn will nudge a model that re-issues the exact same tool
    /// call as the previous round (a repetition loop). Default:
    /// [`MAX_REPEAT_NUDGES`].
    pub max_repeat_nudges: u32,
    /// Max times a turn will silently re-prompt the model to continue after it
    /// stops with text but no tool calls (when it was actively working earlier
    /// in the turn). Keeps the agent going without user intervention. Default:
    /// [`MAX_SILENT_CONTINUES`].
    pub max_silent_continues: u32,
    /// How many times to silently re-run a round that produced no usable output.
    /// Default: [`MAX_EMPTY_RETRIES`].
    pub max_empty_retries: u32,
    /// Max times one turn will nudge the model to continue after its output was
    /// truncated by the output token cap. Separate from `max_empty_retries`
    /// because truncation is a different failure mode (valid output, just cut
    /// off by the token limit) and needs a more generous budget for big tasks.
    /// Default: [`MAX_TRUNCATION_RETRIES`].
    pub max_truncation_retries: u32,
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
    /// Whether read-only review/status/security/gap turns get a deterministic
    /// inspection seed before the first model call. This gives small models
    /// concrete manifests, entrypoints, diffs, and targeted search results to
    /// answer from instead of starting from `list .` only.
    pub read_only_preflight: bool,
    /// Whether long-horizon agency is on: a structured `Goal` the agent
    /// decomposes into sub-goals, drives across turns, retries on failure, and
    /// resumes across sessions. Off by default while it stabilizes; when off,
    /// the agent behaves as the single-turn loop.
    pub long_horizon: bool,
    /// The model id used to decompose a `/goal <objective>` into sub-goals (a
    /// single bounded planning call). `None` disables decomposition — `/goal`
    /// then sets a single sub-goal equal to the objective. Defaults to
    /// `pipe/glm-5.2-fast` on the pipenetwork profile.
    pub planner_model: Option<String>,
    /// The model id used by the `/goal team` skeptic gate to review a turn before
    /// it advances a sub-goal (a bounded critique call). `None` (the default)
    /// disables the gate regardless of `/goal team`. Set via `HI_SKEPTIC_MODEL` or
    /// a profile; a stronger/critical model is the intent (its whole job is to
    /// catch premature "done").
    pub skeptic_model: Option<String>,
    /// When true, ask the user to confirm each file edit (write/edit/multi_edit/
    /// apply_patch) before applying it. The UI shows a diff preview and prompts
    /// for y/n. In non-interactive mode, edits are auto-approved.
    pub confirm_edits: bool,
    /// Workspace-local language-server policy.
    pub lsp_mode: LspMode,
    /// Dynamic, minimal, or full tool advertisement.
    pub tool_set: ToolSet,
    /// Additional project-relative globs omitted from automatic context.
    pub context_exclusions: Vec<String>,
    /// Verifier-gated skill auto-curation: after a turn *passes verification*,
    /// make one tool-free model call to distill any reusable technique from the
    /// turn into a learned skill (`.hi/skills/<slug>/SKILL.md`). The verifier is
    /// the gate, so a weak model can't poison the playbook. Off by default —
    /// costs one extra model call per verified turn; opt in per profile or CLI.
    pub curate_skills: bool,
    /// Advertise the read-only `explore` subagent tool, which lets the model
    /// delegate a bounded read-only investigation to a child agent (own context,
    /// read-only tools, small step budget) and get back a concise answer. The CLI
    /// turns this on by default (it's read-only, depth-capped at 1, and per-session
    /// budgeted); disable per profile with `explore_subagents = false`. Children
    /// never get it (depth ≤ 1).
    pub explore_subagents: bool,
    /// Advertise the write-capable `delegate` subagent tool: the model hands off a
    /// self-contained implementation subtask to a child that can edit + verify in
    /// its own context, and the changes are merged back only if verification passes
    /// (else rolled back to a checkpoint). Off by default — this is the riskier
    /// tier; opt in per profile or via `HI_WRITE_SUBAGENTS`. Depth-capped like
    /// explore (a subagent never gets it).
    pub write_subagents: bool,
    /// True when this agent *is* a subagent (an `explore` or `delegate` child). Set
    /// internally, never by config. It's the depth guard: a subagent is never
    /// advertised the `explore`/`delegate` tools (even in read-only mode), so a
    /// subagent cannot spawn another — capping nesting depth at 1. A top-level
    /// agent (`false`) keeps `explore` even in a read-only/review turn, since
    /// delegating a read-only investigation is itself read-only.
    pub is_subagent: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            workspace_root: std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from(".")),
            state_root: std::path::PathBuf::from(".hi/state"),
            model: String::new(),
            provider_route: None,
            requested_max_tokens: 8192,
            max_tokens: 8192,
            max_tokens_explicit: false,
            temperature: None,
            thinking_budget: None,
            reasoning_effort: None,
            tool_mode: ToolMode::Auto,
            compat: CompatMode::Auto,
            context_window: None,
            project_context: None,
            verification: VerificationMode::Auto,
            max_verify_repairs: 2,
            review: ReviewPolicy::Risk,
            allow_unverified: false,
            allow_no_checkpoint: true,
            max_steps: u32::MAX,
            max_steps_explicit: false,
            auto_compact: true,
            compaction: CompactionKind::ElideThenSummarizeTail {
                keep_recent: DEFAULT_KEEP_RECENT,
            },
            finalize: true,
            max_repeat_nudges: MAX_REPEAT_NUDGES,
            max_silent_continues: MAX_SILENT_CONTINUES,
            max_empty_retries: MAX_EMPTY_RETRIES,
            max_truncation_retries: MAX_TRUNCATION_RETRIES,
            max_parallel_tools: MAX_PARALLEL_TOOLS,
            auto_compact_percent: AUTO_COMPACT_PERCENT,
            compact_target_percent: COMPACT_TARGET_PERCENT,
            in_turn_elide_percent: IN_TURN_ELIDE_PERCENT,
            in_turn_keep_tool_results: IN_TURN_KEEP_TOOL_RESULTS,
            proactive_verify: false,
            read_only_preflight: true,
            long_horizon: false,
            planner_model: None,
            skeptic_model: None,
            confirm_edits: false,
            lsp_mode: LspMode::Auto,
            tool_set: ToolSet::Dynamic,
            context_exclusions: Vec::new(),
            curate_skills: false,
            explore_subagents: false,
            write_subagents: false,
            is_subagent: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_defaults_are_safe_and_automatic() {
        let config = AgentConfig::default();
        assert_eq!(config.verification, VerificationMode::Auto);
        assert_eq!(config.max_verify_repairs, 2);
        assert_eq!(config.review, ReviewPolicy::Risk);
        assert_eq!(config.lsp_mode, LspMode::Auto);
        assert_eq!(config.tool_set, ToolSet::Dynamic);
        assert!(!config.max_steps_explicit);
        assert!(!config.allow_unverified);
        assert!(config.allow_no_checkpoint);
    }
}
