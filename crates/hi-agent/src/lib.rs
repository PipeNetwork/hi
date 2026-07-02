//! The agent loop: user message → model → tool calls → results → repeat
//! until the model stops calling tools. No artificial step limit.

mod agent;
pub mod command;
pub mod compaction;
mod config;
mod decision;
mod goal;
mod heuristics;
mod memory;
mod prompt;
mod session;
mod snapshot;
mod steering;
mod transcript;
pub mod ui;
mod verify;

use std::sync::Arc;

use hi_ai::{Provider, ToolSpec, Usage};

pub use command::Command;
pub use compaction::{CompactionKind, DEFAULT_KEEP_RECENT};
pub use config::{AgentConfig, VerifyStage};
pub use heuristics::humanize_count;
pub use hi_tools::{PlanStatus, PlanStep};
pub use memory::{
    AnnotatedBullet, global_memory_file, memory_file, read_global_memory, read_memory,
    read_project_annotated, should_distill_memory,
};
pub use session::SessionSink;
pub use ui::{Ui, classify_error, tool_label};

use snapshot::SnapshotCache;
use transcript::Transcript;

#[cfg(test)]
use {
    anyhow::Result,
    heuristics::{looks_like_continue, plan_has_pending_steps},
    hi_ai::{Message, ToolMode},
    steering::{
        ConcreteReviewAnswerProblem, EvidenceTracker, ImplementationIntent,
        READ_ONLY_PREFLIGHT_DIFF_MAX_LINES, READ_ONLY_PREFLIGHT_GREP_MAX_LINES, ReviewIntent,
        SecuritySearchFamilies, bounded_review_repair_exhaustion_answer,
        classify_implementation_intent, classify_read_only_intent, compact_preflight_tool_output,
        concrete_review_answer_problem, gpu_training_estimator_bootstrap_files,
        implementation_preflight_command, implementation_turn_prompt,
        implementation_workspace_can_accept_rust_bootstrap_at, inspection_signature,
        insufficient_after_incomplete_security_search, preferred_validation_from_preflight,
        preflight_path_relevant_for_intent, read_only_preflight_initial_calls,
        security_search_families_for_tool, should_nudge_concrete_review_answer,
        should_nudge_security_broad_search, should_nudge_security_scope,
    },
};

pub use decision::{Decision, DecisionLog};
pub use goal::{DEFAULT_SUBGOAL_RETRIES, Goal, GoalStatus, SubGoal};

/// Crate version (from Cargo.toml).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Pre-turn state that must be restored when an attempt is discarded.
///
/// The transcript alone is not enough: tools can update prompt-injected state
/// such as structured goals, plans, and key decisions before the user retries
/// or interrupts the turn.
#[derive(Clone)]
pub struct AgentStateSnapshot {
    pub(crate) goal: Option<String>,
    pub(crate) structured_goal: Option<Goal>,
    pub(crate) decisions: DecisionLog,
    pub(crate) last_plan: Vec<PlanStep>,
}

/// Per-turn telemetry: the trajectory of one `run_turn`, captured so callers
/// (the `--report` writer, the eval harness) can diagnose *how* a turn went,
/// not just whether it passed. The counters here are locals inside `run_turn`
/// that would otherwise be discarded on return; flushing them to this struct
/// makes the verify/recovery/nudge story queryable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnTelemetry {
    /// How many verify rounds ran this turn (0 = verify off or skipped).
    pub verify_rounds: u32,
    /// Times a content-less / malformed response was silently re-sampled
    /// (recovery sampling). 0 on a clean turn.
    pub recovery_retries: u32,
    /// Times the repeat guard nudged the model for re-issuing identical calls.
    pub repeat_nudges: u32,
    /// Times the continue nudge fired for an announced-but-unperformed step.
    pub continue_nudges: u32,
    /// Times the truncation recovery nudged the model to continue after hitting
    /// the output token cap. 0 on a turn that never hit the limit.
    pub truncation_retries: u32,
    /// Whether the turn hit the per-turn step cap (`max_steps`).
    pub hit_step_cap: bool,
    /// Whether the turn ended stalled on announced-but-unrun steps.
    pub stalled_unfinished: bool,
    /// Whether the turn ended stalled repeating the same tool call.
    pub stalled_repeating: bool,
    /// Attributions parsed from the last verify failure's output (empty if
    /// verify passed, was skipped, or produced nothing parseable). Points at
    /// the file/line/symbol the model was steered toward.
    pub verify_attributions: Vec<TurnAttribution>,
    /// Scheduler parallelism this turn: total tool calls executed.
    pub tool_calls: u32,
    /// Largest number of calls that ran concurrently in a single ready-batch
    /// (1 = everything serialized; higher = the dep-aware scheduler overlapped
    /// independent calls). Measures whether the scheduler's concurrency
    /// actually helped.
    pub max_concurrent_batch: u32,
    /// How many calls ran serially (bash, or a lone ready call in a batch).
    /// `tool_calls - serial_runs` is the count that ran as part of a parallel
    /// batch; the parallelism ratio is `(tool_calls - serial_runs) / tool_calls`.
    pub serial_runs: u32,
    /// Per-tool-call timeline for this turn: each call's name, target path,
    /// wall-clock duration (milliseconds), and whether it errored. Ordered by
    /// execution completion. Lets `--report` and the eval harness diagnose
    /// *where* time went and which calls failed, not just aggregate counts.
    pub tool_timeline: Vec<ToolCallEntry>,
    /// Number of successful file-read tool calls this turn.
    pub file_reads: u32,
    /// Number of successful targeted search or diff tool calls this turn.
    pub targeted_searches: u32,
    /// Whether the only successful discovery evidence was a directory listing.
    pub listing_only: bool,
    /// First discovery tool kind observed this turn (`none`, `listing`,
    /// `targeted_search`, or `file_read`).
    pub first_tool_kind: String,
    /// Overall read-only discovery depth (`none`, `listing_only`,
    /// `targeted_search`, `file_read`, or `mixed`).
    pub discovery_depth: String,
    /// Times the harness nudged a read-only review to inspect beyond a listing.
    pub quality_repair_nudges: u32,
}

impl Default for TurnTelemetry {
    fn default() -> Self {
        Self {
            verify_rounds: 0,
            recovery_retries: 0,
            repeat_nudges: 0,
            continue_nudges: 0,
            truncation_retries: 0,
            hit_step_cap: false,
            stalled_unfinished: false,
            stalled_repeating: false,
            verify_attributions: Vec::new(),
            tool_calls: 0,
            max_concurrent_batch: 0,
            serial_runs: 0,
            tool_timeline: Vec::new(),
            file_reads: 0,
            targeted_searches: 0,
            listing_only: false,
            first_tool_kind: "none".to_string(),
            discovery_depth: "none".to_string(),
            quality_repair_nudges: 0,
        }
    }
}

/// A serializable view of one parsed verify-failure location, for the telemetry
/// report. Mirrors `hi_tools::Attribution` but owned and plain-old-data so it
/// derives `Serialize`/`Deserialize` cleanly without leaking the parser type
/// across the crate boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TurnAttribution {
    pub path: String,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub message: String,
    /// `"compile"`, `"test"`, `"lint"`, or `"other"`.
    pub kind: String,
}

impl From<&hi_tools::Attribution> for TurnAttribution {
    fn from(a: &hi_tools::Attribution) -> Self {
        let kind = match a.kind {
            hi_tools::AttrKind::Compile => "compile",
            hi_tools::AttrKind::Test => "test",
            hi_tools::AttrKind::Lint => "lint",
            hi_tools::AttrKind::Other => "other",
        };
        Self {
            path: a.path.clone(),
            line: a.line,
            column: a.column,
            message: a.message.clone(),
            kind: kind.to_string(),
        }
    }
}

/// One entry in the per-turn tool-call timeline: which tool ran, against what
/// path (when inferrable), how long it took, and whether it errored. Lets the
/// `--report` JSON and eval harness diagnose where time went and which calls
/// failed — not just aggregate counts.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolCallEntry {
    /// The tool name (`read`, `write`, `edit`, `bash`, …).
    pub tool: String,
    /// The target path when inferrable (`read`/`write`/`edit` carry one;
    /// `bash` does not). Empty when no single path applies.
    pub path: String,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Whether the tool's output indicated an error (starts with `"Error:"`).
    pub error: bool,
}

/// Auto-compact once the context window is at least this percent full.
pub const AUTO_COMPACT_PERCENT: u64 = 80;
/// After triggering, compact until the local estimate is at or below this
/// percent of the window (so there's headroom before the next compaction).
pub const COMPACT_TARGET_PERCENT: u64 = 50;
/// During one long tool loop, begin dropping old bulky tool payloads before the
/// next model call. This keeps repeated requests from multiplying token spend.
pub const IN_TURN_ELIDE_PERCENT: u64 = 45;
/// Keep the newest tool results verbatim when trimming inside a turn; these are
/// usually the files/errors the model is actively using.
pub const IN_TURN_KEEP_TOOL_RESULTS: usize = 6;
/// User turns auto-compaction keeps verbatim.
pub const AUTO_KEEP_RECENT: usize = 3;
/// How many times to silently re-run a round that produced no usable output —
/// either a content-less response (only reasoning, or empty) or a transient
/// malformed/empty *stream error* — each retry resampling hotter, before giving
/// up and surfacing it.
pub const MAX_EMPTY_RETRIES: u32 = 2;
/// Invalid tool turns from local/open tool models often recover after an explicit
/// schema nudge. Keep this separate from empty/malformed stream retries so normal
/// completion failures do not get a larger budget.
pub const MAX_TOOL_PROTOCOL_RETRIES: u32 = 4;
/// Max times one turn will nudge the model to continue after its output was
/// truncated by the output token cap (`stop_reason: "length"` / `"max_tokens"`).
/// This is a *separate* budget from [`MAX_EMPTY_RETRIES`] because truncation is
/// a different failure mode: the model was producing valid output, it just ran
/// out of token budget. A big task can legitimately hit the cap several times
/// (long file writes, multi-step plans) — sharing the empty-retry budget (only
/// 2) caused the turn to end early on a half-finished output, leaving the model
/// "picking up where it stalled" on the next prompt. A higher, dedicated budget
/// lets the model finish the work without the user having to type "continue".
pub const MAX_TRUNCATION_RETRIES: u32 = 5;
/// Max read-only tool calls to run concurrently within one round, bounding the
/// open file handles / subprocesses a single batched response can spawn.
pub const MAX_PARALLEL_TOOLS: usize = 8;
/// Max times one turn will nudge a model that re-issues the *exact same* tool
/// call as the previous round — a repetition loop where the model re-runs an
/// identical command, gets the same output, and re-emits it again. Bounds the
/// recovery before the turn ends with an honest "stuck repeating" notice;
/// `max_steps` is the hard backstop.
pub const MAX_REPEAT_NUDGES: u32 = 2;
/// Max times a turn will silently re-prompt the model to continue after it
/// stops with text but no tool calls (when it was actively working). Keeps the
/// agent going without user intervention, bounded so it can't loop forever.
/// Set to 5 because some models need 2-3 text-only responses to a nudge before
/// they actually act — with 3, a single step's stall could exhaust the budget
/// and end the turn mid-plan.
pub const MAX_SILENT_CONTINUES: u32 = 5;
/// Maximum number of per-turn git checkpoints retained for `/undo`. Each is a
/// 40-char SHA, so the memory cost is negligible, but a very long session
/// (thousands of turns) would grow the vec without bound. Older checkpoints
/// beyond this cap are dropped — `/undo` only needs the most recent few.
pub const MAX_CHECKPOINTS: usize = 50;
/// Sent silently (no status line, no steer counter) when the model stops with
/// text after having made tool calls earlier in the turn. The system prompt
/// tells the model not to narrate without acting, but when it still does, this
/// keeps the turn going so the user doesn't have to type "continue".
pub(crate) const SILENT_CONTINUE_NUDGE: &str = "Continue now — use your tools to do the work you just \
described. Don't narrate; act. If the task is genuinely complete, stop and give your final recap.";
/// Sent when the model stops calling tools but its plan (posted via `update_plan`)
/// still has pending or active steps. The model often completes one sub-task,
/// writes a recap, and stops — leaving the plan at e.g. 2/9. This nudge points
/// it at the next incomplete step so it keeps working without the user typing
/// "continue".
pub(crate) const PLAN_CONTINUE_NUDGE: &str = "Your plan still has incomplete steps. Continue with the next \
pending step — use your tools to do the work, don't just describe it. Mark the step active in \
`update_plan`, do the work, then move to the next. If the task is genuinely complete, stop and \
give your final recap.";
/// Sent when the model's output was truncated by the output token cap
/// (`stop_reason: "length"` / `"max_tokens"`) — the response was cut off
/// mid-generation, not finished. The nudge tells the model to continue from
/// where it stopped so the turn doesn't end on a half-finished output.
pub(crate) const TRUNCATION_NUDGE: &str = "Your previous response was cut off by the output token limit — \
it was truncated, not finished. Continue from where you stopped, but keep the continuation small: \
finish the current paragraph or call exactly one tool for the next smallest concrete action. Do not \
restart, repeat what you already produced, or write a long narrative continuation.";
pub(crate) const TRUNCATED_TOOL_CALL_NUDGE: &str = "Your previous response was cut off while emitting or preparing a tool \
call. That partial work was not executed. Issue one fresh, complete tool call now. If the payload \
is large, split the work into smaller writes/edits and do only the next chunk; use bounded shell \
smoke tests for verification. Do not continue inside the partial tool-call text or emit prose \
instead of the next concrete action.";

pub(crate) fn partial_text_tool_call_start(text: &str) -> Option<usize> {
    ["<tool_call>", "{\"name\"", "[tool_call", "[tool_calls"]
        .into_iter()
        .filter_map(|marker| text.find(marker))
        .min()
}

/// Asked of the model in a dedicated, tool-free call after a turn that changed
/// files, to guarantee a structured recap even from a model that wouldn't
/// produce one on its own. Kept terse and concrete so weak models still comply.
const FINALIZE_PROMPT: &str = "The work for this turn is done. Write the final summary for the \
user, in past tense, covering only what you actually did:\n\
- One headline line stating what you accomplished.\n\
- A short bullet list of the key changes, grouped by file.\n\
- The exact command(s) to run or test it.\n\
If something is incomplete or a check couldn't run, say so honestly. Output only the summary — \
no preamble, and don't take any further action.";

/// Instruction appended to a slice of history to summarize it for compaction.
const SUMMARIZE_PROMPT: &str = "Summarize our conversation so far into a concise but \
complete handoff brief: the task and goal, key decisions and constraints, files created \
or changed, commands that matter, and any open or next steps. This summary will REPLACE \
the history, so include everything needed to continue seamlessly. Output only the summary.";

const SYSTEM_PROMPT: &str = "\
You are hi, a coding agent running in the user's terminal. Work in the current \
project — modify existing files in place, don't scaffold sub-projects. Prefer \
action over description. Keep responses concise. For non-trivial changes, state \
your plan in one line first. For a task that takes several steps, track it with \
the `update_plan` tool: post the step list up front, then call it again as you \
go — always with the complete list — marking the current step `active` and \
finished ones `done`. Skip the plan for simple one-step changes. Verify your \
edits before finishing. \
\
Never describe a next step without doing it — if you say 'let me read X', call \
the read tool in the same response. Do not narrate intent; act on it. Keep \
working until the task is complete, then stop and give your recap. \
\
Prefer existing project dependencies and standard-library solutions unless the \
user explicitly asks to add a dependency. For generated or new files, keep each \
write/edit small enough to fit comfortably in one tool call; build the file in \
coherent chunks instead of emitting a huge payload. After creating or editing \
code, run a targeted syntax/build/test command before continuing.";

/// Parse an `update_plan` arguments JSON and apply its step statuses to a
/// structured goal's sub-goals (mapping by position). Tolerant — a malformed
/// payload or count mismatch just applies what it can. Used when `long_horizon`
/// is on so the model's stated plan progress drives the goal.
fn apply_plan_to_goal(goal: &mut Goal, arguments: &str) {
    #[derive(serde::Deserialize)]
    struct StepArg {
        #[serde(default)]
        status: String,
    }
    #[derive(serde::Deserialize)]
    struct PlanArgs {
        #[serde(default)]
        steps: Vec<StepArg>,
    }
    if let Ok(args) = serde_json::from_str::<PlanArgs>(arguments) {
        let statuses: Vec<&str> = args.steps.iter().map(|s| s.status.as_str()).collect();
        goal.apply_plan_statuses(&statuses);
    }
}

pub struct Agent {
    pub(crate) provider: Box<dyn Provider>,
    pub(crate) config: AgentConfig,
    /// Conversation history, shared with in-flight `ChatRequest`s via the
    /// `Arc` inside [`Transcript`]. Mutations go through the `Transcript` API
    /// so provider-safety invariants (every `tool_use` has a matching
    /// `tool_result`; typed synthetic nudges) are enforced by construction.
    pub(crate) messages: Transcript,
    pub(crate) tools: Arc<[ToolSpec]>,
    pub(crate) session: Option<Box<dyn SessionSink>>,
    /// How many messages have already been handed to the session sink.
    pub(crate) persisted: usize,
    /// Running total of tokens across the session.
    pub(crate) totals: Usage,
    /// Whether the most recent turn's verification passed (None if not run).
    pub(crate) last_verify: Option<bool>,
    /// Input tokens of the most recent model call — a proxy for how full the
    /// context window is, used to decide when to auto-compact.
    pub(crate) context_used: u64,
    /// Per-turn git checkpoints (working-tree snapshots), for `/undo`.
    pub(crate) checkpoints: Vec<String>,
    /// Files whose content or presence changed in the most recent turn.
    pub(crate) last_changed_files: Vec<String>,
    pub(crate) last_compat_fallbacks: Vec<String>,
    /// A shared interrupt flag. When set (by the UI on a user action like
    /// pressing Esc during a tool call), the agent skips the remaining tool
    /// calls in the current batch and feeds a "interrupted by user" result
    /// back to the model, so it can adapt without losing the turn.
    pub(crate) interrupt: Arc<std::sync::atomic::AtomicBool>,
    /// Telemetry from the most recent `run_turn` (verify rounds, recovery
    /// retries, nudges fired, last verify attributions). Flushed at turn end
    /// from locals that would otherwise be discarded; exposed for `--report`
    /// and the eval harness so they can diagnose *how* a turn went.
    pub(crate) last_turn_telemetry: TurnTelemetry,
    /// Optional transient goal injected into the system prompt for future turns.
    pub(crate) goal: Option<String>,
    /// A structured, multi-step long-horizon goal (decomposed into sub-goals)
    /// used when `config.long_horizon` is on. Persisted across sessions and
    /// injected into the system prompt each turn so the agent resumes the
    /// active sub-goal coherently. Distinct from the transient `goal` string.
    pub(crate) structured_goal: Option<Goal>,
    /// Durable intra-session decision log — recorded via the `record_decision`
    /// tool and injected into the system prompt each turn, so the model stays
    /// consistent across compaction (which would otherwise summarize away the
    /// reasoning behind earlier decisions).
    pub(crate) decisions: DecisionLog,
    /// Cached workspace snapshot — avoids re-walking the tree on every
    /// verify/turn-end check when no files changed. Invalidated by any
    /// write/edit/bash tool call in the current turn, and by `/undo`.
    pub(crate) snapshot_cache: SnapshotCache,
    /// The most recent plan posted via `update_plan` this turn — used to detect
    /// an incomplete plan when the model stops calling tools. If the plan has
    /// pending/active steps, the agent silently nudges the model to continue
    /// rather than ending the turn (the model often writes a finished-looking
    /// recap after one sub-task, even when the plan is only 2/9 done).
    pub(crate) last_plan: Vec<PlanStep>,
}

#[cfg(test)]
mod tests;
