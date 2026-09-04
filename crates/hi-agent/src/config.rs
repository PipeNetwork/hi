//! Per-session agent configuration and the layered-verification stage type.

use hi_ai::{CompatMode, DeepSeekCompat, ReasoningEffort, ToolMode};
use hi_engine_api::EngineMode;
use serde::{Deserialize, Serialize};

use crate::compaction::{CompactionKind, DEFAULT_KEEP_RECENT};
use crate::{
    AUTO_COMPACT_PERCENT, COMPACT_TARGET_PERCENT, IN_TURN_ELIDE_PERCENT, IN_TURN_KEEP_TOOL_RESULTS,
    MAX_EMPTY_RETRIES, MAX_KEEP_WORKING, MAX_MODEL_ROUNDS, MAX_PARALLEL_TOOLS, MAX_REPEAT_NUDGES,
    MAX_SILENT_CONTINUES, MAX_TOOL_CALLS, MAX_TRUNCATION_RETRIES,
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

/// **Completion-review** policy for post-mutation independent / large-diff review.
///
/// This gates [`crate::Agent::independent_review`] / [`crate::Agent::large_diff_review`]
/// after a green workspace verify. It does **not** control Steer-phase
/// **answer-repair** quality nudges ([`ReviewRepairBudgets`]) or the long-horizon
/// **goal-skeptic** gate.
///
/// Prefer the alias [`CompletionReviewPolicy`] in new code for clarity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewPolicy {
    #[default]
    Risk,
    Always,
    Off,
}

/// Alias for [`ReviewPolicy`] — post-mutation completion review only.
pub type CompletionReviewPolicy = ReviewPolicy;

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

/// Negotiation policy for the native Rhai `run_program` capability.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProgramMode {
    /// Advertise the capability only when the provider supports native tools.
    Auto,
    /// Never advertise or execute `run_program`.
    #[default]
    Off,
}

impl ProgramMode {
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Auto)
    }
}

/// Per-session controls for program execution and speculative tool calling.
#[derive(Clone, Debug)]
pub struct AgentProgramConfig {
    pub mode: ProgramMode,
    pub speculative_ptc: bool,
    pub max_speculative_calls: usize,
    pub max_external_speculative_calls: usize,
    pub external_ttl_seconds: u64,
}

/// Configuration for the optional hot-swappable decision engine. The native
/// decision path remains the safe migration default; callers can opt into a
/// validated component with `engine_mode = "wasm"` or `HI_ENGINE_MODE=wasm`.
#[derive(Clone, Debug)]
pub struct AgentEngineConfig {
    pub mode: EngineMode,
    pub module_path: Option<std::path::PathBuf>,
    pub allow_unsigned: bool,
    pub trusted_key_hex: Vec<String>,
    pub watch: bool,
    pub max_guest_fuel: u64,
    pub max_guest_memory_bytes: usize,
    pub max_guest_step_ms: u64,
}

impl Default for AgentEngineConfig {
    fn default() -> Self {
        let mode = std::env::var("HI_ENGINE_MODE")
            .ok()
            .and_then(|value| EngineMode::parse(&value))
            .unwrap_or_default();
        let module_path = std::env::var_os("HI_ENGINE_MODULE").map(std::path::PathBuf::from);
        let allow_unsigned = std::env::var("HI_ENGINE_ALLOW_UNSIGNED")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            });
        let trusted_key_hex = std::env::var("HI_ENGINE_TRUSTED_KEYS")
            .or_else(|_| std::env::var("HI_ENGINE_TRUSTED_KEY"))
            .ok()
            .into_iter()
            .flat_map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect();
        let watch = std::env::var("HI_ENGINE_WATCH").ok().is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        });
        Self {
            mode,
            module_path,
            allow_unsigned,
            trusted_key_hex,
            watch,
            max_guest_fuel: hi_engine_host::DEFAULT_GUEST_FUEL,
            max_guest_memory_bytes: hi_engine_host::DEFAULT_GUEST_MEMORY_BYTES,
            max_guest_step_ms: hi_engine_host::DEFAULT_GUEST_STEP_TIMEOUT_MS,
        }
    }
}

impl Default for AgentProgramConfig {
    fn default() -> Self {
        Self {
            // Keep the new nested execution path opt-in until its accounting
            // and cancellation semantics have accumulated production soak.
            mode: ProgramMode::Off,
            speculative_ptc: false,
            max_speculative_calls: 2,
            max_external_speculative_calls: 1,
            external_ttl_seconds: 30,
        }
    }
}

impl AgentProgramConfig {
    /// Environment kill switches are intentionally narrow and supplemental to
    /// the typed config, which lets operators disable the feature without
    /// rebuilding a long-lived binary.
    pub fn mode_enabled(&self) -> bool {
        if std::env::var("HI_PROGRAM_MODE")
            .ok()
            .is_some_and(|value| value.eq_ignore_ascii_case("off"))
        {
            return false;
        }
        self.mode.is_enabled()
    }

    pub fn speculation_enabled(&self) -> bool {
        self.speculative_ptc
            && !std::env::var("HI_SPECULATIVE_PTC")
                .ok()
                .is_some_and(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "0" | "false" | "off"
                    )
                })
    }
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
    detect_verify_pipeline_with(dir, false)
}

/// Like [`detect_verify_pipeline`], optionally inserting a clippy stage for
/// Cargo workspaces when a public Outcome contract or `--clippy` asks for it.
pub fn detect_verify_pipeline_with(dir: &std::path::Path, clippy: bool) -> Vec<VerifyStage> {
    let has = |name: &str| dir.join(name).exists();
    let stage = |name: &str, command: &str| VerifyStage::new(name, command);
    if has("Cargo.toml") {
        let mut stages = vec![stage("check", "cargo check --quiet")];
        if clippy {
            stages.push(stage("clippy", "cargo clippy --quiet --all-targets"));
        }
        stages.push(stage("test", "cargo test --quiet"));
        stages
    } else if has("go.mod") {
        vec![
            stage("build", "go build ./..."),
            stage("test", "go test ./..."),
        ]
    } else if has("package.json") {
        javascript_pipeline(dir)
    } else if has("pyproject.toml") || has("setup.py") || has("pytest.ini") || has("tox.ini") {
        let mut stages = Vec::new();
        if has("ruff.toml") || has(".ruff.toml") {
            stages.push(stage("lint", "ruff check ."));
        }
        // A Python package marker does not imply a test suite. Pytest exits
        // with status 5 when it collects no tests; treating that as a failed
        // verification stage causes the model to invent tests or otherwise
        // churn the workspace after a valid source-only edit.
        if crate::verify::has_python_tests(dir) {
            stages.push(stage("test", "pytest -q"));
        }
        stages
    } else {
        makefile_pipeline(dir).unwrap_or_default()
    }
}

/// JavaScript/TypeScript pipeline from what the repo actually declares: the
/// lockfile picks the package manager, and only `package.json` scripts that
/// exist become stages — a blind `npm test` in a repo without a test script
/// fails on npm's placeholder and reads as broken verification.
fn javascript_pipeline(dir: &std::path::Path) -> Vec<VerifyStage> {
    let runner = if dir.join("pnpm-lock.yaml").exists() {
        "pnpm"
    } else if dir.join("yarn.lock").exists() {
        "yarn"
    } else if dir.join("bun.lockb").exists() || dir.join("bun.lock").exists() {
        "bun"
    } else {
        "npm"
    };
    let scripts: std::collections::BTreeSet<String> =
        std::fs::read_to_string(dir.join("package.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .and_then(|package| {
                package.get("scripts").and_then(|scripts| {
                    scripts.as_object().map(|map| map.keys().cloned().collect())
                })
            })
            .unwrap_or_default();
    let mut stages = Vec::new();
    if dir.join("tsconfig.json").exists() {
        stages.push(VerifyStage::new(
            "typecheck",
            "npx --no-install tsc --noEmit",
        ));
    } else if scripts.contains("typecheck") {
        stages.push(VerifyStage::new(
            "typecheck",
            format!("{runner} run typecheck"),
        ));
    }
    if scripts.contains("lint") {
        stages.push(VerifyStage::new("lint", format!("{runner} run lint")));
    }
    if scripts.contains("test") {
        let command = match runner {
            "npm" => "npm test --silent".to_string(),
            other => format!("{other} test"),
        };
        stages.push(VerifyStage::new("test", command));
    }
    stages
}

/// `make test` (and `make check`) only when the Makefile declares the target —
/// otherwise `make: *** No rule to make target` masquerades as a code failure.
fn makefile_pipeline(dir: &std::path::Path) -> Option<Vec<VerifyStage>> {
    let makefile = ["Makefile", "makefile"]
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())?;
    let text = std::fs::read_to_string(makefile).ok()?;
    let has_target = |target: &str| {
        text.lines().any(|line| {
            line.strip_prefix(target)
                .is_some_and(|rest| rest.starts_with(':') && !rest.starts_with("::="))
        })
    };
    let mut stages = Vec::new();
    if has_target("check") {
        stages.push(VerifyStage::new("check", "make check"));
    }
    if has_target("test") {
        stages.push(VerifyStage::new("test", "make test"));
    }
    (!stages.is_empty()).then_some(stages)
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Keep the current best-effort turn lifecycle. State is persisted only at
    /// the normal turn settlement boundary when a session sink is installed.
    #[default]
    Ephemeral,
    /// Persist recoverable progress at task boundaries. Durable mode requires
    /// a session sink and writes after the user prompt and each completed tool
    /// batch, so a restart does not discard the task's latest safe point.
    Durable,
}

impl ExecutionMode {
    pub fn is_durable(self) -> bool {
        matches!(self, Self::Durable)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ephemeral => "ephemeral",
            Self::Durable => "durable",
        }
    }
}

#[derive(Clone, Default)]
pub struct AgentConfig {
    /// Whether task progress is checkpointed at durable execution boundaries.
    pub execution: ExecutionMode,
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
    /// Native Rhai program negotiation and speculative execution controls.
    pub program: AgentProgramConfig,
    /// Native/WASM decision engine selection and module limits.
    pub engine: AgentEngineConfig,
    /// Optional RSI control-plane hooks (interactive path stays thin).
    pub rsi: AgentRsi,
    /// Suppress repository-provided executable hooks while constructing the
    /// launch-directory runtime. Startup paths that may restore PipeFS set
    /// this until the remote workspace mode is known.
    pub suppress_initial_project_hooks: bool,
    /// Construct the launch-directory runtime with LSP disabled. A subsequent
    /// controlled workspace rebind still uses `gates.lsp_mode`.
    pub defer_initial_lsp: bool,
    /// Test-only owner for the temporary state root used by unit-test
    /// fixtures. Keeping the guard in the config makes it live as long as the
    /// test agent and removes all state recursively when the last clone drops.
    #[cfg(test)]
    pub(crate) test_state_root: Option<std::sync::Arc<tempfile::TempDir>>,
    /// Test-only owner for a temporary workspace shared by fixtures created
    /// during one test. This lets file and directory fixtures stay outside the
    /// checkout while remaining inside the agent's workspace boundary.
    #[cfg(test)]
    pub(crate) _test_workspace_root: Option<std::sync::Arc<tempfile::TempDir>>,
    /// Test-only explicit process policy. Tests must not mutate `HI_SANDBOX`
    /// because Rust's process environment is shared by all parallel tests.
    #[cfg(test)]
    pub(crate) sandbox_policy: Option<hi_tools::sandbox::SandboxPolicy>,
    /// Per-session ceiling on how many turns the agent may run before it
    /// stops with [`crate::TurnStopReason::TurnLimit`]. `None` (the default)
    /// means **no limit** — the session runs until the user stops it. Set live
    /// with `/turns <n>` (or `/turns off`). Distinct from the per-turn
    /// [`AgentLoopLimits::max_steps`] model-call cap and from a goal's
    /// [`crate::Goal::step_limit`] plan-size cap.
    pub max_turns: Option<u32>,
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
            workspace_root: std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from(".")),
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
    pub top_p: Option<f32>,
    pub output_token_parameter: hi_ai::OutputTokenParameter,
    pub thinking_budget: Option<u32>,
    /// Abstract reasoning level (`reasoning_effort`) applied to every main-turn
    /// request on OpenAI-compatible endpoints that support it; `None` leaves the
    /// endpoint default. See [`hi_ai::ReasoningEffort`]. Housekeeping calls
    /// (compaction/memory/recap) deliberately leave this off. Set via
    /// `--reasoning-effort`, a profile, or `/config reasoning <level>`.
    pub reasoning_effort: Option<ReasoningEffort>,
    pub tool_mode: ToolMode,
    pub compat: CompatMode,
    pub deepseek_compat: DeepSeekCompat,
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
            top_p: None,
            output_token_parameter: hi_ai::OutputTokenParameter::Auto,
            thinking_budget: None,
            reasoning_effort: None,
            tool_mode: ToolMode::default(),
            compat: CompatMode::default(),
            deepseek_compat: DeepSeekCompat::default(),
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
    /// [`crate::UNLIMITED_REPAIR_CYCLES`] is the ordinary default.
    pub max_verify_repairs: u32,
    /// How many times an objected independent/large-diff review may re-enter
    /// Model for repair before the turn stalls as [`crate::ReviewStatus::Objected`].
    /// `0` means the first objection is final (no repair cycle), while
    /// [`crate::UNLIMITED_REPAIR_CYCLES`] allows productive repairs until the
    /// review passes or a no-progress/fault circuit fires.
    pub max_independent_review_repairs: u32,
    /// **Completion-review** policy ([`ReviewPolicy`] / [`CompletionReviewPolicy`]).
    /// Does not disable Steer answer-repair or goal-skeptic.
    pub review: ReviewPolicy,
    /// Permit a mutation turn to complete with `Unverified` status.
    pub allow_unverified: bool,
    /// Legacy compatibility setting. Reviewer transport/protocol failures no
    /// longer block deterministically verified productive goal work; they are
    /// always recorded as unavailable diagnostics. Concrete reviewer objections
    /// still block regardless of this value.
    pub skeptic_fail_open: bool,
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
    /// When true, print planned tool actions without executing them (dry run).
    /// Mutating calls report what they *would* do; nothing touches the
    /// workspace or spawns processes.
    pub dry_run: bool,
    /// Workspace-local language-server policy.
    pub lsp_mode: LspMode,
}

impl Default for AgentGates {
    fn default() -> Self {
        Self {
            verification: VerificationMode::Auto,
            max_verify_repairs: crate::UNLIMITED_REPAIR_CYCLES,
            max_independent_review_repairs: crate::UNLIMITED_REPAIR_CYCLES,
            review: ReviewPolicy::Risk,
            allow_unverified: false,
            // Retained for configuration/source compatibility. Review transports
            // record Unavailable without re-entering Model or invalidating
            // deterministic verification.
            skeptic_fail_open: false,
            allow_no_checkpoint: true,
            proactive_verify: false,
            read_only_preflight: true,
            confirm_edits: false,
            dry_run: false,
            lsp_mode: LspMode::Auto,
        }
    }
}

/// Human-readable form of a productive-repair limit. Keep the internal
/// `u32::MAX` sentinel out of user-facing status and diagnostic text.
pub(crate) fn repair_limit_label(limit: u32) -> String {
    if limit == crate::UNLIMITED_REPAIR_CYCLES {
        "unlimited".to_string()
    } else {
        limit.to_string()
    }
}

/// Human-readable remaining productive repairs under `limit`.
pub(crate) fn remaining_repairs_label(limit: u32, used: u32) -> String {
    if limit == crate::UNLIMITED_REPAIR_CYCLES {
        "unlimited".to_string()
    } else {
        limit.saturating_sub(used).to_string()
    }
}

/// Whether another productive repair may run. The all-ones value is an
/// unlimited sentinel, not a very large finite count, so a saturated usage
/// counter must never make the ordinary default look exhausted.
pub(crate) fn repair_available(limit: u32, used: u32) -> bool {
    limit == crate::UNLIMITED_REPAIR_CYCLES || used < limit
}

pub(crate) fn repair_limit_reached(limit: u32, used: u32) -> bool {
    !repair_available(limit, used)
}

/// Caps that bound a single turn's model/tool loops.
#[derive(Clone, Debug)]
pub struct AgentLoopLimits {
    /// Optional **hard** wall-clock budget through the turn's settlement commit:
    /// active model/tool work is cancelled when it expires, cancellation
    /// cleanup is allowed to settle, and `run_turn` returns a deadline error.
    /// If the body already committed, the deadline only cancels best-effort
    /// terminal notices and the committed result is returned. Prefer
    /// [`Self::turn_soft_deadline`] and keep this as a backstop.
    pub turn_timeout: Option<std::time::Duration>,
    /// Optional **soft** wall-clock budget. Unlike [`Self::turn_timeout`] this
    /// never interrupts work in flight: once it expires the loop simply stops
    /// starting new model/repair rounds and proceeds to Settle → Finalize, so
    /// the turn ends with its workspace reconciled, its report written, and an
    /// honest `TurnStopReason::TimeLimit`.
    ///
    /// Set it below whatever external deadline the caller faces (CI job, bench
    /// harness, wrapper timeout). Being killed at that deadline instead makes
    /// the result a lottery on whatever happened to be on disk mid-edit.
    pub turn_soft_deadline: Option<std::time::Duration>,
    /// Cap on model calls per turn. [`MAX_MODEL_ROUNDS`] (`u32::MAX`) means
    /// unlimited and is the ordinary default. Set a finite cap via
    /// `--max-steps`, `/config steps <n>`, or an internal subagent budget. A
    /// capped turn gets one tool-free wrap-up round, then proceeds through
    /// normal workspace verification and settlement.
    pub max_steps: u32,
    /// Cap on executed tool calls per turn. [`MAX_TOOL_CALLS`] (`u32::MAX`)
    /// means unlimited and is the ordinary default. This is independent of the
    /// model-call (`max_steps`) cap.
    pub max_tool_calls: u32,
    /// Max times one turn will nudge a model that re-issues the exact same tool
    /// call as the previous round (a repetition loop). Default:
    /// [`MAX_REPEAT_NUDGES`].
    pub max_repeat_nudges: u32,
    /// Max times a turn will silently re-prompt after a text-only response
    /// while a structured plan or goal still has work. Ordinary prose is not
    /// a continuation signal. Default: [`MAX_SILENT_CONTINUES`].
    pub max_silent_continues: u32,
    /// Extra recoveries after a stall budget is spent, so the agent keeps
    /// working instead of asking the user to `/retry`. Default:
    /// [`MAX_KEEP_WORKING`]. `0` disables the recovery (tests).
    pub max_keep_working: u32,
    /// How many times to silently re-run a round that produced no usable output.
    /// Default: [`MAX_EMPTY_RETRIES`].
    pub max_empty_retries: u32,
    /// Max times one turn will nudge the model to continue after its output was
    /// truncated by the output token cap. [`MAX_TRUNCATION_RETRIES`]
    /// (`u32::MAX`) is the ordinary unlimited default; finite values are an
    /// explicit integration policy.
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
            // Ordinary productive work has no whole-turn wall-clock ceiling.
            // Frontends and harnesses can still install explicit soft/hard
            // deadlines; individual provider/tool operations retain their own
            // fault-containment timeouts.
            turn_timeout: None,
            turn_soft_deadline: None,
            max_steps: MAX_MODEL_ROUNDS,
            max_tool_calls: MAX_TOOL_CALLS,
            max_repeat_nudges: MAX_REPEAT_NUDGES,
            max_silent_continues: MAX_SILENT_CONTINUES,
            max_keep_working: MAX_KEEP_WORKING,
            max_empty_retries: MAX_EMPTY_RETRIES,
            max_truncation_retries: MAX_TRUNCATION_RETRIES,
            max_parallel_tools: MAX_PARALLEL_TOOLS,
            review_repair: ReviewRepairBudgets::default(),
        }
    }
}

impl AgentLoopLimits {
    /// Whether an explicitly finite tool-execution cap has been reached.
    /// `u32::MAX` is an unlimited sentinel, not a reachable cap.
    pub(crate) fn tool_call_cap_reached(&self, calls: u32) -> bool {
        self.max_tool_calls != u32::MAX && calls >= self.max_tool_calls
    }

    /// Remaining executable tools under the current cap. Preserve the
    /// unlimited sentinel instead of turning it into a very large, decreasing
    /// finite budget as calls accrue.
    pub(crate) fn remaining_tool_calls(&self, calls: u32) -> u32 {
        if self.max_tool_calls == u32::MAX {
            u32::MAX
        } else {
            self.max_tool_calls.saturating_sub(calls)
        }
    }

    /// Whether another valid output-truncation continuation is available.
    /// Preserve `u32::MAX` as an unlimited sentinel even after the diagnostic
    /// counter itself saturates; a direct `<` comparison would otherwise turn
    /// saturation into a false cap.
    pub(crate) fn truncation_retry_available(&self, retries: u32) -> bool {
        self.max_truncation_retries == u32::MAX || retries < self.max_truncation_retries
    }
}

/// How many times each review-answer repair mode may fire in one turn.
///
/// Defaults match the historical hard-coded mode limits. Operators can lower
/// them for cheaper/stricter sessions or raise them for stubborn models.
#[derive(Clone, Debug, PartialEq, Eq)]
/// Per-mode budgets for **answer-repair** quality nudges (Steer phase).
///
/// Distinct from [`AgentGates::max_verify_repairs`] (workspace shell) and
/// [`AgentGates::max_independent_review_repairs`] (completion-review Object cycles).
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
    /// Force a chat-only answer after inspection-sprawl already fired and the
    /// model still tries to continue inspecting. Separate from cascade quality
    /// spends so earlier answer-repairs cannot starve this path.
    pub sprawl_force_answer: u32,
}

/// Alias clarifying that these budgets are Steer **answer-repair**, not
/// completion-review or workspace verify.
pub type AnswerRepairBudgets = ReviewRepairBudgets;

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
            sprawl_force_answer: 3,
        }
    }
}

impl ReviewRepairBudgets {
    /// Budget for a stable answer-repair mode key (`review_no_evidence`, …).
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
            "review_sprawl_force_answer" => self.sprawl_force_answer,
            // Unknown keys: no budget (was silent default 2). Callers should
            // only pass ReviewRepairMode::key() values.
            _ => 0,
        }
    }
}

/// Context window management, project context, and tool catalog selection.
#[derive(Clone, Debug)]
pub struct AgentMemory {
    /// Project context (e.g. from HI.md/AGENTS.md) appended to the system prompt.
    pub project_context: Option<String>,
    /// User standing rules from `~/.config/hi/me.md`, also in the stable system
    /// prompt. Session-constant so the prefix cache stays hot.
    pub standing_rules: Option<String>,
    /// When the context window fills past a threshold, summarize-and-reset
    /// before the next turn so a long session doesn't overflow the model.
    pub auto_compact: bool,
    /// Strategy used by `/compact` (no arg) and the summarizing tier of
    /// auto-compaction.
    pub compaction: CompactionKind,
    /// After a mutating turn — or a tool-using turn that produced no visible
    /// answer — make one dedicated tool-free model call to produce a recap.
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
    /// Census-driven per-project trim (`hi tools trim`): tool names removed
    /// from advertisement after the usage census showed them dead. Names on
    /// the protected floor ([`hi_tools::PROTECTED_TOOLS`]) are ignored here
    /// even if present, so a wrong or corrupted list can cost tokens but
    /// never capability.
    pub disabled_tools: Vec<String>,
    /// Glob-style path exclusions applied when ranking repository context.
    pub context_exclusions: Vec<String>,
    /// Whether the agent may curate/learn skills during the session.
    pub curate_skills: bool,
    /// Read from and append to the durable failure-learning ledger. This is
    /// independent from skill curation, but follows the session persistence
    /// policy: `--no-memory` and `--no-save` turn it off together with the
    /// other durable learning surfaces.
    pub learning: bool,
    /// Whether the matching stack pack (rust-workspace / pytest-package /
    /// ts-monorepo) is auto-injected into the per-turn volatile context block.
    /// On by default in production; the test harness disables it so canned-
    /// provider tests measure stable token budgets and message shapes.
    pub inject_stack_skill: bool,
    /// Whether the `code-review` pack is auto-injected on review-shaped turns.
    /// On by default; the test harness disables it like [`Self::inject_stack_skill`].
    pub inject_review_skill: bool,
    /// After a successful turn, predict a Claude-style "suggested next prompt"
    /// for the interactive input bar (ghost text). Side call; off for
    /// subagents / plan mode / goal auto-drive regardless of this flag.
    pub suggest_next_prompt: bool,
    /// Advertise `ask_user` (and honor it). This is opt-in: autonomous coding
    /// must not park indefinitely on a clarification that it can resolve from
    /// repository evidence or a safe default.
    pub offer_ask_user: bool,
    /// Advertise `search_tool` / `use_tool` (two gateway schemas, not each MCP
    /// tool's schema). Set only after at least one `.hi/mcp` server connects.
    pub offer_mcp: bool,
    /// Advertise inject-gated markdown memory tools (`memory_search` / `get` /
    /// `update` / `forget`). Off for `--no-memory` / `--no-save`.
    pub offer_memory: bool,
    /// Advertise inject-gated `browser_exec` when the task looks like page/login/UI.
    /// On by default; `[browser] enabled = false` turns it off.
    pub offer_browser: bool,
    /// Allow `browser_exec` to navigate RFC1918 / loopback hosts.
    pub browser_allow_private: bool,
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self {
            project_context: None,
            standing_rules: None,
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
            disabled_tools: Vec::new(),
            context_exclusions: Vec::new(),
            curate_skills: false,
            learning: true,
            inject_stack_skill: true,
            inject_review_skill: true,
            // On by default for interactive coding; disable via profile / env /
            // `/config suggest off`.
            suggest_next_prompt: true,
            offer_ask_user: false,
            offer_mcp: false,
            offer_memory: false,
            offer_browser: true,
            browser_allow_private: false,
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
    /// Model id for write-capable `delegate` executors (`None` = the driver's
    /// model). With `delegate_endpoint`, lets a big cloud driver dispatch
    /// execution to a local model (team roles: big brain plans, local hands
    /// type).
    pub delegate_model: Option<String>,
    /// Optional OpenAI-compatible base URL for delegate executors only.
    pub delegate_endpoint: Option<String>,
    /// API key sent to `delegate_endpoint`.
    pub delegate_endpoint_key: Option<String>,
    /// Model id for read-only `explore` recon children (`None` = the driver's
    /// model). The `HI_EXPLORE_MODEL` env var still wins when set.
    pub explore_model: Option<String>,
    /// Optional OpenAI-compatible base URL for explore children only. When
    /// unset, explore children share the driver's provider connection.
    pub explore_endpoint: Option<String>,
    /// API key sent to `explore_endpoint`.
    pub explore_endpoint_key: Option<String>,
    /// Model id for `delegate` calls tagged `kind: "edit"` — mechanical,
    /// precisely-specified changes. Team-bench showed small fast models win
    /// that task shape (nemotron-4b: 36 tok/s, passes edit/json) while only
    /// big coders author reliably, so the two lanes are routable separately.
    /// `None` = edits ride the normal delegate route.
    pub editor_model: Option<String>,
    /// Optional OpenAI-compatible base URL for the editor lane only.
    pub editor_endpoint: Option<String>,
    /// API key sent to `editor_endpoint`.
    pub editor_endpoint_key: Option<String>,
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
            delegate_model: None,
            delegate_endpoint: None,
            delegate_endpoint_key: None,
            explore_model: None,
            explore_endpoint: None,
            explore_endpoint_key: None,
            editor_model: None,
            editor_endpoint: None,
            editor_endpoint_key: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_defaults_are_safe_and_automatic() {
        let config = AgentConfig::default();
        assert_eq!(config.gates.verification, VerificationMode::Auto);
        assert_eq!(
            config.gates.max_verify_repairs,
            crate::UNLIMITED_REPAIR_CYCLES
        );
        assert_eq!(
            config.gates.max_independent_review_repairs,
            crate::UNLIMITED_REPAIR_CYCLES
        );
        assert_eq!(
            repair_limit_label(config.gates.max_verify_repairs),
            "unlimited"
        );
        assert_eq!(
            remaining_repairs_label(config.gates.max_verify_repairs, 42),
            "unlimited"
        );
        assert!(repair_available(crate::UNLIMITED_REPAIR_CYCLES, u32::MAX));
        assert!(!repair_limit_reached(
            crate::UNLIMITED_REPAIR_CYCLES,
            u32::MAX
        ));
        assert!(!repair_available(3, 3));
        assert!(repair_limit_reached(3, 3));
        assert_eq!(config.gates.review, ReviewPolicy::Risk);
        assert_eq!(config.gates.lsp_mode, LspMode::Auto);
        assert_eq!(config.memory.tool_set, ToolSet::Dynamic);
        assert_eq!(config.loop_limits.max_steps, u32::MAX);
        assert_eq!(config.loop_limits.max_tool_calls, u32::MAX);
        assert_eq!(config.loop_limits.max_truncation_retries, u32::MAX);
        assert_eq!(config.loop_limits.turn_soft_deadline, None);
        assert_eq!(config.loop_limits.turn_timeout, None);
        assert!(!config.loop_limits.tool_call_cap_reached(u32::MAX));
        assert!(
            config.loop_limits.truncation_retry_available(u32::MAX),
            "a saturated diagnostic counter must not become a false unlimited cap"
        );
        assert_eq!(
            config.loop_limits.remaining_tool_calls(48),
            u32::MAX,
            "ordinary tool execution is unlimited rather than a decreasing finite budget"
        );
        assert!(!config.gates.allow_unverified);
        assert!(config.gates.allow_no_checkpoint);
        assert!(config.subagents.explore_subagents, "explore on by default");
        assert_eq!(config.subagents.write_subagents, WriteSubagentPolicy::Risk);
        assert_eq!(config.program.mode, ProgramMode::Off);
        assert!(!config.program.mode_enabled());
        assert!(!config.program.speculation_enabled());
        assert_eq!(config.program.max_speculative_calls, 2);
        assert_eq!(config.program.max_external_speculative_calls, 1);
        assert_eq!(config.program.external_ttl_seconds, 30);
        let budgets = &config.loop_limits.review_repair;
        assert_eq!(budgets.no_evidence, 4);
        assert_eq!(budgets.read_after_search, 2);
        assert_eq!(budgets.security_scope, 5);
        assert_eq!(budgets.gap_search_overclaim, 3);
        assert_eq!(budgets.sprawl_force_answer, 3);
        assert_eq!(budgets.limit_for_key("review_listing_only"), 4);
        assert_eq!(budgets.limit_for_key("review_sprawl_force_answer"), 3);
        // Typos must not silently get budget 2 (old fail-open default).
        assert_eq!(budgets.limit_for_key("review_typo_mode"), 0);
    }

    #[test]
    fn explicit_tool_call_cap_remains_finite() {
        let mut limits = AgentLoopLimits {
            max_tool_calls: 48,
            ..AgentLoopLimits::default()
        };

        assert_eq!(limits.remaining_tool_calls(47), 1);
        assert!(!limits.tool_call_cap_reached(47));
        assert_eq!(limits.remaining_tool_calls(48), 0);
        assert!(limits.tool_call_cap_reached(48));

        limits.max_tool_calls = u32::MAX;
        assert_eq!(limits.remaining_tool_calls(u32::MAX), u32::MAX);
        assert!(!limits.tool_call_cap_reached(u32::MAX));
    }

    #[test]
    fn explicit_truncation_retry_cap_remains_finite() {
        let limits = AgentLoopLimits {
            max_truncation_retries: 5,
            ..AgentLoopLimits::default()
        };

        assert!(limits.truncation_retry_available(4));
        assert!(!limits.truncation_retry_available(5));
        assert!(!limits.truncation_retry_available(u32::MAX));
    }

    #[test]
    fn cargo_clippy_pipeline_inserts_clippy_between_check_and_test() {
        let dir = std::env::temp_dir().join(format!(
            "hi-clippy-pipeline-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname=\"x\"\nversion=\"0.1.0\"\n",
        )
        .unwrap();
        let names: Vec<_> = detect_verify_pipeline_with(&dir, true)
            .into_iter()
            .map(|stage| stage.name)
            .collect();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(names, ["check", "clippy", "test"]);
    }
}
