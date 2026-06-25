//! The agent loop: user message → model → tool calls → results → repeat
//! until the model stops calling tools. No artificial step limit.

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
mod transcript;
mod verify;
pub mod ui;

use std::sync::Arc;

use anyhow::Result;
use futures_util::StreamExt;
use hi_ai::{
    ChatRequest, Content, Message, Provider, ProviderErrorKind, RequestProfile,
    StreamEvent, ToolMode, ToolSpec, Usage, provider_error_kind, provider_error_usage,
};
use hi_tools::{TOOL_SPECS, execute, execute_streaming};

pub use command::Command;
pub use compaction::{CompactionKind, DEFAULT_KEEP_RECENT};
pub use config::{AgentConfig, VerifyStage};
pub use heuristics::humanize_count;
pub use hi_tools::{PlanStatus, PlanStep};
pub use memory::{memory_file, should_distill_memory};
pub use session::SessionSink;
pub use ui::{Ui, tool_label};

use heuristics::{
    emit_tool_output, looks_like_unfinished_step, looks_mutating, recovery_sampling,
    recovery_telemetry, respects_deps, tool_deps, tool_mode_label, StallMode, RECOVERY_SAMPLING,
};
use memory::{cap_memory, memory_prompt};
use prompt::SystemPrompt;
use snapshot::{changed_files_between, FileFingerprint, SnapshotCache};
use transcript::{NudgeKind, Transcript};
use verify::{stage_guidance, Verifier, VerifyOutcome, Snapshot};

pub use decision::{Decision, DecisionLog};
pub use goal::{Goal, GoalStatus, SubGoal, DEFAULT_SUBGOAL_RETRIES};

/// Crate version (from Cargo.toml).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Per-turn telemetry: the trajectory of one `run_turn`, captured so callers
/// (the `--report` writer, the eval harness) can diagnose *how* a turn went,
/// not just whether it passed. The counters here are locals inside `run_turn`
/// that would otherwise be discarded on return; flushing them to this struct
/// makes the verify/recovery/nudge story queryable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
/// Max read-only tool calls to run concurrently within one round, bounding the
/// open file handles / subprocesses a single batched response can spawn.
pub const MAX_PARALLEL_TOOLS: usize = 8;
/// Max times one turn will nudge a model that stopped after *describing* a next
/// step without performing it. Bounds the "narrate-then-stall" recovery;
/// `max_steps` is the hard backstop.
pub const MAX_CONTINUE_NUDGES: u32 = 2;
/// Max times one turn will nudge a model that re-issues the *exact same* tool
/// call as the previous round — a repetition loop where the model re-runs an
/// identical command, gets the same output, and re-emits it again. Bounds the
/// recovery before the turn ends with an honest "stuck repeating" notice;
/// `max_steps` is the hard backstop.
pub const MAX_REPEAT_NUDGES: u32 = 2;
/// Sent when the model announces a next step but emits no tool call, to get it to
/// actually do the work instead of ending the turn on a false "done".
const CONTINUE_NUDGE: &str = "You described a next step but didn't run it. Continue now, \
using your tools, to actually make that change. If the task is genuinely already complete, \
stop and give your final recap: what you changed (by file) and the exact command to run or test it.";

/// Sent when the model re-issues the exact same tool call as the previous
/// round. The command already ran and its output is in the history just above —
/// re-running it will only produce the same result. This nudges the model to act
/// on that output (edit the code, move on, or finish) instead of looping.
const REPEAT_NUDGE: &str = "You just ran that exact command last round and its output is already \
in the conversation above — running it again will only repeat the same result. Act on that output \
now: make the edit it points to, move to the next step, or if the task is already complete, stop \
and give your final recap. Do not re-run the same command.";

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
edits before finishing.";

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
    provider: Box<dyn Provider>,
    config: AgentConfig,
    /// Conversation history, shared with in-flight `ChatRequest`s via the
    /// `Arc` inside [`Transcript`]. Mutations go through the `Transcript` API
    /// so provider-safety invariants (every `tool_use` has a matching
    /// `tool_result`; typed synthetic nudges) are enforced by construction.
    messages: Transcript,
    tools: Arc<[ToolSpec]>,
    session: Option<Box<dyn SessionSink>>,
    /// How many messages have already been handed to the session sink.
    persisted: usize,
    /// Running total of tokens across the session.
    totals: Usage,
    /// Running USD cost. `None` means some usage was recorded while pricing was
    /// unknown, so showing a precise total would be misleading.
    cost_usd: Option<f64>,
    /// Whether the most recent turn's verification passed (None if not run).
    last_verify: Option<bool>,
    /// Input tokens of the most recent model call — a proxy for how full the
    /// context window is, used to decide when to auto-compact.
    context_used: u64,
    /// Per-turn git checkpoints (working-tree snapshots), for `/undo`.
    checkpoints: Vec<String>,
    /// Files whose content or presence changed in the most recent turn.
    last_changed_files: Vec<String>,
    last_compat_fallbacks: Vec<String>,
    /// Telemetry from the most recent `run_turn` (verify rounds, recovery
    /// retries, nudges fired, last verify attributions). Flushed at turn end
    /// from locals that would otherwise be discarded; exposed for `--report`
    /// and the eval harness so they can diagnose *how* a turn went.
    last_turn_telemetry: TurnTelemetry,
    /// Optional transient goal injected into the system prompt for future turns.
    goal: Option<String>,
    /// A structured, multi-step long-horizon goal (decomposed into sub-goals)
    /// used when `config.long_horizon` is on. Persisted across sessions and
    /// injected into the system prompt each turn so the agent resumes the
    /// active sub-goal coherently. Distinct from the transient `goal` string.
    structured_goal: Option<Goal>,
    /// Durable intra-session decision log — recorded via the `record_decision`
    /// tool and injected into the system prompt each turn, so the model stays
    /// consistent across compaction (which would otherwise summarize away the
    /// reasoning behind earlier decisions).
    decisions: DecisionLog,
    /// Cached workspace snapshot — avoids re-walking the tree on every
    /// verify/turn-end check when no files changed. Invalidated by any
    /// write/edit/bash tool call in the current turn, and by `/undo`.
    snapshot_cache: SnapshotCache,
}

impl Agent {
    /// Start a fresh session seeded with the system prompt.
    pub fn new(provider: Box<dyn Provider>, config: AgentConfig) -> Self {
        let system = SystemPrompt::new()
            .with_project_context(config.project_context.as_deref())
            .with_finalize(config.finalize)
            .build();
        Self::with_messages(provider, config, vec![system], 0)
    }

    /// Resume from previously-saved history (which already includes the system
    /// prompt). The loaded messages are treated as already persisted.
    pub fn resume(
        provider: Box<dyn Provider>,
        config: AgentConfig,
        history: Vec<Message>,
        usage: Usage,
        cost_usd: Option<f64>,
    ) -> Self {
        let persisted = history.len();
        let mut agent = Self::with_messages(provider, config, history, persisted);
        agent.totals = usage;
        agent.cost_usd = cost_usd;
        agent
    }

    fn with_messages(
        provider: Box<dyn Provider>,
        config: AgentConfig,
        messages: Vec<Message>,
        persisted: usize,
    ) -> Self {
        Self {
            provider,
            config,
            messages: Transcript::new(messages),
            tools: TOOL_SPECS.clone().into(),
            session: None,
            persisted,
            totals: Usage::default(),
            cost_usd: Some(0.0),
            last_verify: None,
            context_used: 0,
            checkpoints: Vec::new(),
            last_changed_files: Vec::new(),
            last_compat_fallbacks: Vec::new(),
            last_turn_telemetry: TurnTelemetry::default(),
            goal: None,
            structured_goal: None,
            decisions: DecisionLog::default(),
            snapshot_cache: SnapshotCache::default(),
        }
    }

    /// Revert the file changes the most recent turn made, restoring its git
    /// checkpoint. Returns `None` if there's nothing to undo, else the number of
    /// files restored or removed.
    pub async fn undo(&mut self) -> Result<Option<usize>> {
        let Some(target) = self.checkpoints.pop() else {
            return Ok(None);
        };
        let n = hi_tools::checkpoint::restore(std::path::Path::new("."), &target).await?;
        // The working tree just changed under us, so any cached snapshot is now
        // stale. Without this, the next turn reuses pre-undo fingerprints and
        // change detection / verify gating / last_changed_files can be wrong.
        self.invalidate_snapshot();
        Ok(Some(n))
    }

    /// Attach a session sink that records messages produced from here on.
    pub fn set_session(&mut self, session: Box<dyn SessionSink>) {
        self.session = Some(session);
    }

    /// The current conversation history (including the system prompt).
    pub fn messages(&self) -> &[Message] {
        self.messages.as_slice()
    }

    /// Discard messages back to `len` — used to drop an interrupted turn so the
    /// conversation stays consistent (no dangling user message, no orphan
    /// tool_use from a round cut off mid-execution).
    pub fn truncate_messages(&mut self, len: usize) {
        self.messages.rewind_to(len);
        self.persisted = self.persisted.min(self.messages.len());
    }

    /// Cumulative token usage across the session.
    pub fn totals(&self) -> &Usage {
        &self.totals
    }

    /// Cumulative USD cost across the session, if pricing is known.
    pub fn cost_usd(&self) -> Option<f64> {
        self.cost_usd
    }

    /// The context-window occupancy, as last reported by the provider.
    pub fn context_used(&self) -> u64 {
        self.context_used
    }

    /// The configured context window, if known.
    pub fn context_window(&self) -> Option<u32> {
        self.config.context_window
    }

    /// Render the conversation as Markdown for /export.
    pub fn export_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# hi session transcript

");
        for msg in self.messages.as_slice().iter() {
            match msg.role {
                hi_ai::Role::System => {} // skip system prompt
                hi_ai::Role::User => {
                    out.push_str("**user:**

");
                    out.push_str(&msg.text());
                    out.push_str("

");
                }
                hi_ai::Role::Assistant => {
                    out.push_str("**assistant:**\n\n");
                    out.push_str(&msg.text());
                    out.push_str("\n\n");
                }
                hi_ai::Role::Tool => {
                    out.push_str("**tool:**\n\n");
                    out.push_str(&msg.text());
                    out.push_str("\n\n");
                }
            }
        }
        out
    }

    fn add_usage(&mut self, usage: Usage) {
        if !usage.is_zero() {
            match (self.cost_usd, self.config.price) {
                (Some(total), Some((input_price, output_price))) => {
                    // Prefer the provider-computed normalized breakdown when
                    // available — it correctly decomposes tokens into priced
                    // buckets regardless of provider (OpenAI's prompt_tokens
                    // includes cached; Anthropic reports cache separately), so
                    // a session that switches models mid-run still accrues cost
                    // coherently. Fall back to the input-excludes-cache
                    // heuristic for legacy/error paths that don't set `billable`.
                    let cost = if let Some(b) = usage.billable {
                        (b.regular_input as f64 * input_price
                            + b.cached_input as f64 * input_price * 0.5
                            + b.cache_creation as f64 * input_price * 1.25
                            + b.output as f64 * output_price)
                            / 1_000_000.0
                    } else {
                        let regular_input =
                            usage.input_tokens.saturating_sub(usage.cache_read_tokens);
                        (regular_input as f64 * input_price
                            + usage.cache_read_tokens as f64 * input_price * 0.5
                            + usage.cache_creation_tokens as f64 * input_price * 1.25
                            + usage.output_tokens as f64 * output_price)
                            / 1_000_000.0
                    };
                    self.cost_usd = Some(total + cost);
                }
                (_, None) => {
                    self.cost_usd = None;
                }
                (None, Some(_)) => {}
            }
        }
        self.totals.add(usage);
        let effective_input = usage.effective_input_tokens();
        if effective_input > 0 {
            self.context_used = effective_input;
        }
    }

    fn add_error_usage(&mut self, err: &anyhow::Error) {
        self.add_usage(provider_error_usage(err));
    }

    /// Number of git checkpoints created so far (for `/undo`).
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// The model id currently configured for this session.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Switch the model used for subsequent turns, refreshing the pricing and
    /// context-window metadata that drive the usage display.
    pub fn set_model(
        &mut self,
        model: String,
        price: Option<(f64, f64)>,
        context_window: Option<u32>,
    ) {
        self.config.model = model;
        self.config.price = price;
        self.config.context_window = context_window;
    }

    /// Reset the live context to just the system prompt. This is transient: it
    /// doesn't rewrite the session file, and the reset point isn't persisted, so
    /// resuming replays the full log.
    pub fn clear_history(&mut self) {
        self.messages.replace_all(vec![self.system_message()]);
        self.persisted = self.messages.len();
    }

    fn system_message(&self) -> Message {
        let goal_section = self
            .structured_goal
            .as_ref()
            .and_then(|g| g.prompt_section());
        SystemPrompt::new()
            .with_project_context(self.config.project_context.as_deref())
            .with_goal(self.goal.as_deref())
            .with_goal_state(goal_section.as_deref())
            .with_decisions(self.decisions.prompt_section().as_deref())
            .with_finalize(self.config.finalize)
            .build()
    }

    /// Minimal system message for throwaway model calls (finalize_turn,
    /// summarize, update_memory) — no project_context, no goal, no finalize
    /// instruction. These calls don't need the repo map or session goal; sending
    /// them wastes ~1.5-3K input tokens per call and bloats the uncached portion
    /// of the request.
    fn minimal_system_message(&self) -> Message {
        SystemPrompt::new().build()
    }

    fn refresh_system_message(&mut self) {
        let system = self.system_message();
        self.messages.replace_system(system);
    }

    /// Current transient session goal, if any.
    pub fn goal(&self) -> Option<&str> {
        self.goal.as_deref()
    }

    /// The durable intra-session decision log (recorded via `record_decision`),
    /// injected into the system prompt each turn and preserved across compaction.
    pub fn decisions(&self) -> &DecisionLog {
        &self.decisions
    }

    /// Set or clear the transient session goal and inject it into the system prompt.
    pub fn set_goal(&mut self, goal: Option<String>) {
        self.goal = goal.and_then(|g| {
            let g = g.trim().to_string();
            (!g.is_empty()).then_some(g)
        });
        self.refresh_system_message();
    }

    /// Set or clear a structured long-horizon goal (decomposed into sub-goals).
    /// Only takes effect when `config.long_horizon` is on; when set, the goal's
    /// state is injected into the system prompt each turn so the agent resumes
    /// the active sub-goal. Returns whether it was accepted.
    pub fn set_structured_goal(&mut self, goal: Option<Goal>) -> bool {
        if !self.config.long_horizon && goal.is_some() {
            return false;
        }
        self.structured_goal = goal;
        self.refresh_system_message();
        // Persist the change so a /resume picks it up. Best-effort: no UI here
        // (callers can surface failures), so swallow — the goal lives in-memory
        // for this session regardless.
        if let Some(session) = self.session.as_mut()
            && let Some(g) = &self.structured_goal
        {
            let _ = session.record_goal(g);
        }
        true
    }

    /// The structured long-horizon goal, if any (for persistence/observability).
    pub fn structured_goal(&self) -> Option<&Goal> {
        self.structured_goal.as_ref()
    }

    /// Whether long-horizon agency is on (the `long_horizon` config flag), so
    /// frontends can branch `/goal` between the structured goal and the
    /// transient goal string.
    pub fn long_horizon(&self) -> bool {
        self.config.long_horizon
    }

    /// Long-horizon driver — called at turn end. When a structured goal is set
    /// and `long_horizon` is on, advance or retry the active sub-goal based on
    /// the turn's outcome, so the next turn resumes at the right sub-goal (with
    /// prior-attempt notes if it stalled, so the model doesn't repeat a failed
    /// approach). The verify retry itself happens *within* the turn (the 'turn
    /// loop re-runs the model on a verify failure); this hook handles the
    /// goal-level progression once the turn settles.
    fn goal_turn_end(
        &mut self,
        stalled_unfinished: bool,
        stalled_repeating: bool,
        hit_step_cap: bool,
        ui: &mut dyn Ui,
    ) {
        if !self.config.long_horizon {
            return;
        }
        let Some(goal) = self.structured_goal.as_mut() else {
            return;
        };
        if goal.status != GoalStatus::Active {
            return; // Already done or failed — nothing to drive.
        }
        let max_retries = DEFAULT_SUBGOAL_RETRIES;
        // A turn that verified clean (or had no verify and didn't stall)
        // completes the active sub-goal → advance.
        let verified_clean = matches!(self.last_verify, Some(true));
        let no_verify_clean = self.last_verify.is_none() && !stalled_unfinished && !stalled_repeating && !hit_step_cap;
        if verified_clean || no_verify_clean {
            let i = goal.active_index();
            goal.advance();
            if let Some(i) = i {
                ui.status(&format!(
                    "✓ sub-goal {}/{} done — advancing",
                    i + 1,
                    goal.sub_goals.len().max(i + 1)
                ));
            }
            if goal.status == GoalStatus::Done {
                ui.status("✓ long-horizon goal complete");
            }
            self.refresh_system_message();
            self.persist_goal(ui);
            return;
        }
        // A stalled or cap-hit turn, or a verify failure that ended the turn,
        // records a sub-goal attempt so the next turn sees the prior note. If
        // the budget is exhausted, the sub-goal (and goal) is marked Failed.
        let reason = if hit_step_cap {
            "hit the per-turn step cap"
        } else if stalled_unfinished {
            "stalled announcing steps without running them"
        } else if stalled_repeating {
            "stalled repeating the same tool call"
        } else {
            "verification failed and the turn ended without fixing it"
        };
        let can_retry = goal.record_failure(reason, max_retries);
        if can_retry {
            ui.status(&format!(
                "↻ sub-goal failed this turn ({reason}) — will retry next turn, don't repeat the same approach"
            ));
        } else {
            ui.status(&format!(
                "✗ sub-goal exhausted its retry budget ({reason}) — marked failed; /goal to revise or continue past it"
            ));
        }
        self.refresh_system_message();
        self.persist_goal(ui);
    }

    /// Handle a `record_decision` tool call: parse the args, append to the
    /// durable decision log (which feeds the system prompt), and return a
    /// terse confirmation for the model. Malformed args yield an error string
    /// (the model sees it and can retry), not a panic.
    fn handle_record_decision(&mut self, arguments: &str) -> String {
        #[derive(serde::Deserialize)]
        struct DecisionArgs {
            summary: String,
            rationale: String,
            #[serde(default)]
            files: Vec<String>,
        }
        match serde_json::from_str::<DecisionArgs>(arguments) {
            Ok(args) => {
                let summary = args.summary.trim().to_string();
                if summary.is_empty() {
                    return "Error: record_decision needs a non-empty summary".to_string();
                }
                self.decisions.record(Decision {
                    summary,
                    rationale: args.rationale.trim().to_string(),
                    files: args.files,
                });
                // Refresh the system prompt so the decision is injected on the
                // next turn (and visible to the model immediately in history).
                self.refresh_system_message();
                "Decision recorded — it will persist across compaction.".to_string()
            }
            Err(err) => format!("Error: bad record_decision arguments: {err}"),
        }
    }

    /// Whether the most recent turn's verification passed (None if not run).
    pub fn last_verify(&self) -> Option<bool> {
        self.last_verify
    }

    /// Files whose content or presence changed in the most recent turn.
    pub fn last_changed_files(&self) -> &[String] {
        &self.last_changed_files
    }

    /// Compatibility fallbacks that were triggered in the most recent turn.
    pub fn last_compat_fallbacks(&self) -> &[String] {
        &self.last_compat_fallbacks
    }

    /// Telemetry from the most recent turn: verify rounds, recovery retries,
    /// nudges fired, stall flags, and the attributions parsed from the last
    /// verify failure. Lets callers diagnose *how* a turn went, not just
    /// whether it passed.
    pub fn last_turn_telemetry(&self) -> &TurnTelemetry {
        &self.last_turn_telemetry
    }

    /// The tool mode currently configured for this session.
    pub fn tool_mode(&self) -> ToolMode {
        self.config.tool_mode
    }

    /// Whether any verification stage is configured.
    pub fn verify_is_on(&self) -> bool {
        !self.config.verify.is_empty()
    }

    /// A one-line summary of the verification pipeline (`"off"` when none) —
    /// e.g. `"cargo check → cargo test"`.
    pub fn verify_summary(&self) -> String {
        if self.config.verify.is_empty() {
            "off".to_string()
        } else {
            self.config
                .verify
                .iter()
                .map(|s| s.command.as_str())
                .collect::<Vec<_>>()
                .join(" → ")
        }
    }

    /// The models the current provider/endpoint actually serves (via its
    /// `/models` route), with any live metadata — for the `/model` picker and
    /// the live context/price/health wiring. Empty if unsupported.
    pub async fn list_models(&self) -> Result<Vec<hi_ai::ServedModel>> {
        self.provider.list_models().await
    }

    /// Set or clear a single custom verify command (from `/verify <cmd>`),
    /// replacing any configured pipeline with one stage (or clearing it).
    pub fn set_verify_command(&mut self, cmd: Option<String>) {
        self.config.verify = match cmd {
            Some(c) => vec![VerifyStage::new("verify", c)],
            None => Vec::new(),
        };
    }

    /// Replace the verification pipeline (from auto-detection).
    pub fn set_verify_pipeline(&mut self, stages: Vec<VerifyStage>) {
        self.config.verify = stages;
    }

    /// The compaction strategy configured for this session.
    pub fn compaction_kind(&self) -> CompactionKind {
        self.config.compaction.clone()
    }

    /// Reclaim context using the session's configured strategy. Transient like
    /// [`clear_history`](Self::clear_history): the session file keeps the full
    /// log, so resuming replays everything.
    pub async fn compact(&mut self, ui: &mut dyn Ui) -> Result<()> {
        self.compact_with(self.config.compaction.clone(), ui).await
    }

    /// Reclaim context using a specific strategy (e.g. `/compact <kind>`).
    pub async fn compact_with(&mut self, kind: CompactionKind, ui: &mut dyn Ui) -> Result<()> {
        match kind {
            CompactionKind::Summarize => self.compact_summarize(ui).await,
            CompactionKind::Hybrid { keep_recent } => self.compact_hybrid(keep_recent, ui).await,
            CompactionKind::ElideToolOutput { keep_recent } => {
                self.compact_elide(keep_recent, ui);
                Ok(())
            }
            CompactionKind::ElideThenSummarizeTail { keep_recent } => {
                self.compact_elide_then_summarize_tail(keep_recent, ui).await
            }
        }
    }

    /// Provider byte/request caps can be lower than the model catalog's token
    /// window, so a request can be rejected before usage is reported and before
    /// the normal auto-compaction trigger fires. Keep the latest user request,
    /// drop earlier in-memory context once, and let the loop retry immediately.
    fn retry_after_request_too_large(
        &mut self,
        input: &str,
        turn_start: usize,
        ui: &mut dyn Ui,
    ) -> bool {
        if turn_start <= 1 {
            return false;
        }

        self.truncate_messages(1);
        self.messages.push_user(format!(
            "[Earlier conversation context was omitted because the provider rejected the request \
             as too large. Continue from this latest user request; ask for missing details if the \
             omitted context is required.]\n\n{input}"
        ));
        self.context_used = 0;
        ui.status(
            "provider rejected the request as too large; dropped prior conversation context and retrying",
        );
        true
    }

    /// Summarize the whole conversation and reset to system + summary.
    async fn compact_summarize(&mut self, ui: &mut dyn Ui) -> Result<()> {
        // Need at least one exchange beyond the system prompt to summarize.
        if self.messages.len() <= 1 {
            ui.status("nothing to compact yet");
            return Ok(());
        }
        // Own the slice so it doesn't borrow `self` across the `&mut self` call.
        let slice = self.messages.as_slice()[1..].to_vec();
        let Some(summary) = self.summarize(&slice, ui).await? else {
            ui.status("compaction produced no summary; keeping history");
            return Ok(());
        };
        let system = self.system_message();
        self.messages.replace_all(vec![
            system,
            Message::user(format!("[Summary of the conversation so far]\n\n{summary}")),
        ]);
        self.persisted = self.messages.len();
        // Persist the compaction so it survives session resume.
        if let Some(session) = self.session.as_mut() {
            let _ = session.record_compaction(&self.messages.arc());
        }
        ui.status("✓ compacted — context reset to the summary");
        Ok(())
    }

    /// Keep the last `keep_recent` user turns verbatim; summarize everything
    /// older and fold the brief into the first kept turn. Folding (rather than
    /// inserting a separate summary message) avoids two consecutive user
    /// messages, which some providers reject.
    async fn compact_hybrid(&mut self, keep_recent: usize, ui: &mut dyn Ui) -> Result<()> {
        let Some(split) = compaction::recent_split(self.messages.as_slice(), keep_recent) else {
            // Nothing older than the recent window — summarize everything so a
            // triggered compaction still makes progress.
            return self.compact_summarize(ui).await;
        };
        let old = self.messages.as_slice()[1..split].to_vec();
        let Some(summary) = self.summarize(&old, ui).await? else {
            ui.status("compaction produced no summary; keeping history");
            return Ok(());
        };

        let system = self.system_message();
        let mut recent = self.messages.as_slice()[split..].to_vec();
        let head = recent[0].text();
        recent[0] = Message::user(format!(
            "[Summary of earlier conversation]\n\n{summary}\n\n---\n\n{head}"
        ));
        let mut next = Vec::with_capacity(recent.len() + 1);
        next.push(system);
        next.extend(recent);
        self.messages.replace_all(next);
        self.persisted = self.messages.len();
        // Persist the compaction so it survives session resume.
        if let Some(session) = self.session.as_mut() {
            let _ = session.record_compaction(&self.messages.arc());
        }
        ui.status("✓ compacted — kept recent turns, summarized the rest");
        Ok(())
    }

    /// Elide-first, summarize-only-the-conversational-tail. Keep the recent
    /// `keep_recent` turns verbatim (their tool results elided, skeleton kept).
    /// For old turns: **keep** the tool-bearing ones in history with their bulky
    /// output elided (the call/result skeleton stays, so the model remembers
    /// "I read file X" — just without the verbatim output), and summarize only
    /// the tool-free Q&A turns into a brief folded into the first kept turn. A
    /// pure tool-heavy session with no old Q&A makes no model call at all — just
    /// the deterministic elision.
    async fn compact_elide_then_summarize_tail(
        &mut self,
        keep_recent: usize,
        ui: &mut dyn Ui,
    ) -> Result<()> {
        let Some(split) = compaction::recent_split(self.messages.as_slice(), keep_recent) else {
            // Nothing older than the recent window — fall back to summarizing
            // everything so a triggered compaction still makes progress.
            return self.compact_summarize(ui).await;
        };
        // Elide bulky tool output everywhere older than the recent window. The
        // tool-bearing messages themselves stay (skeleton kept); only their
        // output is stubbed.
        compaction::elide_tool_outputs(self.messages.mutate_slice(), split);

        // Summarize only the conversational (tool-free) old tail. The tool-bearing
        // old turns are NOT summarized — they stay in history, elided.
        let convo = compaction::conversational_tail(self.messages.as_slice(), split);
        let summary = if convo.is_empty() {
            None
        } else {
            self.summarize(&convo, ui).await?
        };

        // Rebuild: system + old tool-bearing turns (elided, kept) + recent turns
        // (with the Q&A summary folded into the first recent turn). The old
        // Q&A-only messages are dropped (replaced by the summary).
        let system = self.system_message();
        let old = self.messages.as_slice()[1..split]
            .iter()
            .filter(|m| {
                m.content
                    .iter()
                    .any(|c| matches!(c, Content::ToolCall { .. } | Content::ToolResult { .. }))
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut recent = self.messages.as_slice()[split..].to_vec();
        let had_summary = summary.is_some();
        if let Some(summary) = summary {
            // Fold the brief into the first kept turn (avoids two consecutive
            // user messages, which some providers reject) — same shape as
            // `compact_hybrid`. If the old tool-bearing region is non-empty, the
            // summary sits between it and the recent turns as a user message —
            // which is fine as long as it doesn't create two consecutive users.
            // The last old message is a ToolResult (tool-bearing), so a user
            // summary after it alternates correctly.
            let head = recent[0].text();
            recent[0] = Message::user(format!(
                "[Summary of earlier conversation]\n\n{summary}\n\n---\n\n{head}"
            ));
        }
        let mut next = Vec::with_capacity(1 + old.len() + recent.len());
        next.push(system);
        next.extend(old);
        next.extend(recent);
        self.messages.replace_all(next);
        self.persisted = self.messages.len();
        if let Some(session) = self.session.as_mut() {
            let _ = session.record_compaction(&self.messages.arc());
        }
        if had_summary {
            ui.status("✓ compacted — elided old tool output, summarized the Q&A tail");
        } else {
            ui.status("✓ compacted — elided old tool output (no Q&A tail to summarize)");
        }
        Ok(())
    }

    /// Deterministically shrink the bulky output of old tool calls. No model
    /// call. Mutates already-persisted messages in place; the session file keeps
    /// the originals, so this stays transient.
    fn compact_elide(&mut self, keep_recent: usize, ui: &mut dyn Ui) {
        // Only turns older than the recent window are eligible; if everything is
        // recent there's nothing to elide.
        let freed = match compaction::recent_split(self.messages.as_slice(), keep_recent) {
            Some(split) => compaction::elide_tool_outputs(self.messages.mutate_slice(), split),
            None => 0,
        };
        if freed > 0 {
            ui.status(&format!(
                "✓ elided ~{}k chars of old tool output",
                freed / 1000
            ));
        } else {
            ui.status("nothing old to elide");
        }
    }

    fn elide_in_turn_context_if_needed(&mut self, ui: &mut dyn Ui) {
        if !self.config.auto_compact {
            return;
        }
        let Some(window) = self.config.context_window else {
            return;
        };
        if window == 0 {
            return;
        }

        let used = compaction::estimate_tokens(self.messages.as_slice());
        if used * 100 < u64::from(window) * self.config.in_turn_elide_percent {
            return;
        }

        let freed = compaction::elide_tool_outputs_except_recent(
            self.messages.mutate_slice(),
            self.config.in_turn_keep_tool_results,
        );
        if freed == 0 {
            return;
        }

        ui.status(&format!(
            "context ~{}% full — elided old tool output before continuing",
            used * 100 / u64::from(window)
        ));
        self.context_used = 0;
    }

    /// Run the summarization model call over `slice`, returning the summary text
    /// (trimmed), or `None` if the model produced nothing. Shared by the
    /// Summarize and Hybrid strategies.
    async fn summarize(&mut self, slice: &[Message], ui: &mut dyn Ui) -> Result<Option<String>> {
        ui.status("compacting the conversation…");

        // Elide bulky tool outputs before sending to the model — the summary
        // doesn't need verbatim command output, just the conversation shape.
        // This can cut input tokens by 50-80% on tool-heavy sessions.
        let mut slice_owned: Vec<Message> = slice.to_vec();
        let len = slice_owned.len();
        compaction::elide_tool_outputs(&mut slice_owned, len);

        let mut messages = Vec::with_capacity(slice_owned.len() + 2);
        messages.push(self.minimal_system_message());
        messages.extend_from_slice(&slice_owned);
        messages.push(Message::user(SUMMARIZE_PROMPT));

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages: Arc::from(messages),
            tools: Arc::new([]), // summarizing — no tool use
            max_tokens: 1024, // throwaway call — summaries are short
            temperature: self.config.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            profile: RequestProfile {
                compat: self.config.compat,
                tool_mode: ToolMode::ChatOnly,
                stream_usage: None,
            },
        };

        let mut summary = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                summary.push_str(&text);
                ui.assistant_text(&text);
            }
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Reasoning(_) => {}
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(completion) => completion,
            Err(err) => {
                self.add_error_usage(&err);
                let _ = self.persist();
                return Err(err);
            }
        };
        ui.assistant_end();
        self.add_usage(completion.usage);
        let _ = self.persist();
        ui.usage(
            self.totals.input_tokens,
            self.totals.output_tokens,
            self.context_used,
            self.config.context_window,
        );

        // Fall back to the final content if the provider didn't stream text.
        if summary.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(t) = c {
                    summary.push_str(t);
                }
            }
        }
        let summary = summary.trim();
        Ok((!summary.is_empty()).then(|| summary.to_string()))
    }

    /// Distill durable, reusable lessons from this session into the project memory
    /// file (`.hi/memory.md`), then load it as context next session. Re-derives the
    /// *whole* capped list from the current memory + this session, so stale or wrong
    /// facts fall out instead of accreting (self-correcting against poisoning).
    ///
    /// One chat-only model call. Best-effort: a provider/IO error is surfaced as a
    /// status, never fatal (it runs at quit). Like [`summarize`](Self::summarize) it
    /// builds a throwaway message vec and does NOT record into the session history.
    pub async fn update_memory(&mut self, ui: &mut dyn Ui) {
        self.update_memory_at(memory_file(), ui).await;
    }

    /// See [`update_memory`](Self::update_memory); the path is a parameter so tests
    /// can redirect the write to a temp file (no global env/cwd state).
    async fn update_memory_at(&mut self, path: std::path::PathBuf, ui: &mut dyn Ui) {
        let existing = std::fs::read_to_string(&path).unwrap_or_default();

        ui.status("distilling session memory…");

        // Elide bulky tool outputs — the memory distillation only needs to
        // understand what was done, not re-read command output verbatim.
        let start = self.messages.len().min(1);
        let mut history: Vec<Message> = self.messages.as_slice()[start..].to_vec();
        let len = history.len();
        compaction::elide_tool_outputs(&mut history, len);

        let mut messages = Vec::with_capacity(history.len() + 2);
        messages.push(self.minimal_system_message());
        messages.extend_from_slice(&history);
        messages.push(Message::user(memory_prompt(existing.trim())));

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages: Arc::from(messages),
            tools: Arc::new([]), // distilling — no tool use
            max_tokens: 1024, // throwaway call — memory notes are short
            temperature: self.config.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            profile: RequestProfile {
                compat: self.config.compat,
                tool_mode: ToolMode::ChatOnly,
                stream_usage: None,
            },
        };

        let mut memory = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                memory.push_str(&text);
                ui.assistant_text(&text);
            }
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Reasoning(_) => {}
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(completion) => completion,
            Err(err) => {
                self.add_error_usage(&err);
                let _ = self.persist();
                ui.status(&format!("(couldn't update memory: {err})"));
                return;
            }
        };
        ui.assistant_end();
        self.add_usage(completion.usage);
        let _ = self.persist();

        // Fall back to the final content if the provider didn't stream text.
        if memory.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(t) = c {
                    memory.push_str(t);
                }
            }
        }
        let memory = cap_memory(&memory);
        if memory.is_empty() {
            return; // nothing durable to save
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let notes = memory.lines().filter(|l| !l.trim().is_empty()).count();
        match std::fs::write(&path, format!("{memory}\n")) {
            Ok(()) => ui.status(&format!(
                "✓ saved {notes} memory note(s) to {}",
                path.display()
            )),
            Err(err) => ui.status(&format!("(couldn't write memory: {err})")),
        }
    }

    /// Get the workspace snapshot, using the cached version when available.
    /// The cache is valid until invalidated by [`invalidate_snapshot`].
    async fn snapshot_cached(&mut self) -> std::collections::BTreeMap<String, FileFingerprint> {
        self.snapshot_cache.get().await
    }

    /// Invalidate the snapshot cache — called after any mutating tool.
    fn invalidate_snapshot(&mut self) {
        self.snapshot_cache.invalidate();
    }

    /// Run one user turn to completion, emitting output through `ui`.
    ///
    /// After the model stops calling tools, an optional verification command is
    /// run; if it fails, its output is fed back and the model iterates, up to
    /// `max_verify_iterations` rounds.
    pub async fn run_turn(&mut self, input: &str, ui: &mut dyn Ui) -> Result<()> {
        if self.tools_unavailable_for(input) {
            ui.status(&format!(
                "tool mode {} does not allow file edits or shell commands for this turn",
                tool_mode_label(self.config.tool_mode)
            ));
            self.messages.push_user(input);
            self.messages.push_assistant(vec![Content::Text(format!(
                "I cannot perform coding actions in {} mode because file-edit and shell tools are unavailable. Switch to `--tool-mode auto` or `--tool-mode required` to let me modify the workspace.",
                tool_mode_label(self.config.tool_mode)
            ))]);
            ui.turn_end(&self.usage_summary(&self.totals));
            self.persist()?;
            return Ok(());
        }
        // Snapshot the working tree before this turn touches anything, so `/undo`
        // can revert it. Best-effort: no-op outside a git repo.
        if let Some(sha) = hi_tools::checkpoint::create(std::path::Path::new(".")).await {
            self.checkpoints.push(sha);
        }

        // If the context window is filling up, reclaim room before adding more,
        // so the session keeps going instead of overflowing. Two tiers: a free,
        // deterministic elision of old tool output first; then, only if still
        // heavy, the configured summarizing strategy. Best-effort — a failed
        // model call just leaves the (already elided) history as-is.
        //
        // The outer trigger uses the provider-reported `context_used` (the last
        // request's occupancy — the most accurate signal, and only meaningful
        // once a real request has happened, so a fresh session isn't
        // over-eagerly compacted). Tier 2 below gates on a local token estimate
        // instead, because `context_used` is stale by then.
        if self.config.auto_compact
            && let Some(window) = self.config.context_window
            && window > 0
            && self.context_used * 100 >= u64::from(window) * self.config.auto_compact_percent
        {
            ui.status(&format!(
                "context ~{}% full — compacting to free room",
                self.context_used * 100 / u64::from(window)
            ));
            // Tier 1: deterministic, no model call. Only old turns are eligible.
            if let Some(split) =
                compaction::recent_split(self.messages.as_slice(), AUTO_KEEP_RECENT)
            {
                compaction::elide_tool_outputs(self.messages.mutate_slice(), split);
            }
            // Tier 2: only if still heavy. `context_used` reflects the
            // pre-elision request and is now stale, so gate on a local estimate.
            let target = u64::from(window) * self.config.compact_target_percent / 100;
            if compaction::estimate_tokens(self.messages.as_slice()) > target {
                let _ = self.compact(ui).await;
            }
            self.context_used = 0;
        }

        let turn_start = self.messages.len();
        self.messages.push_user_or_fold(input);
        self.last_verify = None;
        self.last_changed_files.clear();
        self.last_compat_fallbacks.clear();
        let mut compat_fallbacks = Vec::new();

        let mut verifier = Verifier::new(self.config.verify.clone(), self.config.max_verify_iterations);
        let max_steps = self.config.max_steps;
        let mut steps = 0u32;
        let mut empty_retries = 0u32;
        let mut continue_nudges = 0u32;
        let mut repeat_nudges = 0u32;
        // Scheduler parallelism counters: how many calls ran this turn, the
        // largest concurrent ready-batch, and how many ran serially (bash or a
        // lone ready call). Flushed into telemetry so the dep-aware scheduler's
        // concurrency is measurable, not shipped on faith.
        let mut sched_tool_calls = 0u32;
        let mut sched_max_concurrent = 0u32;
        let mut sched_serial_runs = 0u32;
        // Signature (name, arguments) of the previous round's tool calls, to
        // spot a model re-issuing the exact same call and looping on it.
        let mut prev_call_sig: Option<Vec<(String, String)>> = None;
        let mut request_too_large_retried = false;
        // Whether the model has run a tool this turn (so a later text-only stop is
        // eligible for a continue-nudge) and whether the turn ended on an
        // announced-but-unperformed step (drives the incomplete notice).
        let mut made_tool_call = false;
        let mut stalled_unfinished = false;
        // Whether the turn ended because the model kept re-issuing the exact
        // same tool call through the whole repeat-nudge budget (drives the
        // incomplete notice and skips the finalization recap).
        let mut stalled_repeating = false;
        // Whether the turn was cut short by the per-turn step cap, so the
        // finalization recap is skipped (the work may be incomplete).
        let mut ended_at_cap = false;
        // Attributions parsed from the most recent verify failure — captured
        // here so they survive to turn end and can be flushed into telemetry.
        let mut last_verify_attributions: Vec<hi_tools::Attribution> = Vec::new();
        // Snapshot the turn baseline so verification only runs when the
        // workspace ends up changed. This catches `bash` edits too, while
        // skipping verify when a turn makes no net file changes.
        let turn_snapshot: Snapshot = self.snapshot_cached().await;
        // Snapshot from the most recent verify check. Reused at turn end to
        // avoid a second full tree walk when verify already took one.
        let mut verify_snapshot: Option<Snapshot> = None;

        'turn: loop {
            // Inner loop: model + tools until the model stops calling tools, or
            // the per-turn step cap is hit.
            let hit_cap = loop {
                if steps >= max_steps {
                    break true;
                }
                steps += 1;

                // After a content-less/garbled round, resample hotter and with
                // nucleus + frequency penalty on the retry to break out of the
                // low-entropy attractor that produced it (cf. minion's recovery
                // sampling). Bounded, and only while consecutively stalling —
                // `empty_retries` resets on real output, so a normal round runs at
                // the configured sampling. Toggleable via HI_RECOVERY_SAMPLING for
                // A/B-ing on the eval harness.
                let (temperature, top_p, frequency_penalty) =
                    recovery_sampling(empty_retries, self.config.temperature, *RECOVERY_SAMPLING);

                // Telemetry for the recovery-sampling A/B: emit a concise debug
                // line only when sampling is actually being changed (recovery on
                // and this is a retry), so ordinary runs stay quiet. The empty
                // path is the only mode that escalates sampling today; repeat and
                // continue nudges re-run at the configured sampling.
                if let Some(line) = recovery_telemetry(
                    StallMode::Empty,
                    empty_retries,
                    self.config.max_empty_retries,
                    temperature,
                    top_p,
                    frequency_penalty,
                    *RECOVERY_SAMPLING,
                ) {
                    ui.status(&line);
                }

                self.elide_in_turn_context_if_needed(ui);

                // Debug-mode invariant check: the transcript we're about to send
                // must be provider-safe (every tool_use answered, no consecutive
                // user messages). Cheap in release builds; in debug it catches
                // the orphan-tool_use class of bug at the source.
                debug_assert!(
                    self.messages.validate_for_provider().is_ok(),
                    "transcript invariant violated before provider send"
                );

                let request = ChatRequest {
                    model: self.config.model.clone(),
                    messages: self.messages.arc(),
                    tools: self.request_tools(),
                    max_tokens: self.config.max_tokens,
                    temperature,
                    top_p,
                    frequency_penalty,
                    thinking_budget: self.config.thinking_budget,
                    profile: RequestProfile {
                        compat: self.config.compat,
                        tool_mode: self.config.tool_mode,
                        stream_usage: None,
                    },
                };

                let mut sink = |event: StreamEvent| match event {
                    StreamEvent::Text(text) => ui.assistant_text(&text),
                    StreamEvent::Reasoning(text) => ui.assistant_reasoning(&text),
                    StreamEvent::Status(text) => {
                        if let Some(fallback) = text.strip_prefix("compat: ") {
                            compat_fallbacks.push(fallback.to_string());
                        }
                        ui.status(&text);
                    }
                };
                let mut completion = match self.provider.stream(request, &mut sink).await {
                    Ok(completion) => completion,
                    Err(err)
                        if provider_error_kind(&err)
                            == Some(ProviderErrorKind::RequestTooLarge) =>
                    {
                        if !request_too_large_retried
                            && self.retry_after_request_too_large(input, turn_start, ui)
                        {
                            request_too_large_retried = true;
                            continue;
                        }
                        self.truncate_messages(turn_start);
                        ui.status(
                            "request still exceeds the provider limit with prior context removed; \
                             shorten the prompt or attached input, then retry",
                        );
                        self.add_error_usage(&err);
                        let _ = self.persist();
                        return Err(err);
                    }
                    // A transient generation flake — a malformed/garbled stream or
                    // an empty completion. Treat it like a content-less response:
                    // flush, then silently re-run with hotter recovery sampling (a
                    // fresh request, with its own transport retries) up to the same
                    // budget, instead of failing the turn. Terminal errors (auth,
                    // outage, …) fall through to the abort below.
                    Err(err)
                        if empty_retries < self.config.max_empty_retries
                            && matches!(
                                provider_error_kind(&err),
                                Some(
                                    ProviderErrorKind::MalformedStream
                                        | ProviderErrorKind::EmptyCompletion
                                )
                            ) =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        empty_retries += 1;
                        ui.status(&format!(
                            "⚠ the model's response didn't come through cleanly — \
                             retrying ({empty_retries}/{})",
                            self.config.max_empty_retries
                        ));
                        continue;
                    }
                    Err(err) => {
                        self.add_error_usage(&err);
                        let _ = self.persist();
                        return Err(err);
                    }
                };
                ui.assistant_end();

                self.add_usage(completion.usage);
                // Let the frontend show the running total climb mid-turn.
                ui.usage(
                    self.totals.input_tokens,
                    self.totals.output_tokens,
                    self.context_used,
                    self.config.context_window,
                );

                let calls: Vec<(String, String, String)> = completion
                    .tool_calls()
                    .into_iter()
                    .map(|c| {
                        (
                            c.id.to_string(),
                            c.name.to_string(),
                            c.arguments.to_string(),
                        )
                    })
                    .collect();

                // Repetition guard: the model re-issued the exact same tool
                // calls (same names, same arguments, same order) as the previous
                // round. Re-running them can only reproduce the same output, so
                // don't execute — nudge the model to act on the output it already
                // has. Bounded; past the budget the turn ends with an honest
                // "stuck repeating" notice rather than looping until `max_steps`.
                let call_sig: Vec<(String, String)> = calls
                    .iter()
                    .map(|(_, name, args)| (name.clone(), args.clone()))
                    .collect();
                let is_repeat = !calls.is_empty() && prev_call_sig.as_ref() == Some(&call_sig);
                if is_repeat {
                    // Record this round's assistant text (the model did emit
                    // something) before nudging, so the history stays coherent.
                    // We deliberately do NOT execute the repeated tool calls, so
                    // strip their `ToolCall` blocks from the recorded message:
                    // `push_assistant_text_only` is the intentional "calls
                    // skipped, not executed" path — leaving `tool_use` blocks
                    // without matching `tool_result` blocks puts the transcript
                    // in a state most providers reject on the next request.
                    self.messages.push_assistant_text_only(std::mem::take(&mut completion.content));
                    if repeat_nudges < self.config.max_repeat_nudges {
                        repeat_nudges += 1;
                        stalled_repeating = true;
                        ui.status(&format!(
                            "the model re-ran the same command — its output is already above; \
                             nudging it to act on it ({repeat_nudges}/{})",
                            self.config.max_repeat_nudges
                        ));
                        self.messages.push_nudge(NudgeKind::Repeat, REPEAT_NUDGE);
                        // Keep prev_call_sig as-is so a further repeat is still
                        // detected against the same signature.
                        continue;
                    }
                    ui.status(
                        "⚠ the model kept re-running the same command without acting on the \
                         result — the task may be incomplete. /retry, or send 'continue'.",
                    );
                    break false;
                }
                // A different set of calls (or none) this round — the model moved
                // on, so clear any pending repeat-stall state.
                stalled_repeating = false;
                prev_call_sig = Some(call_sig);

                // This round's assistant text, joined and captured before the
                // content is moved into history. Used both to detect a content-less
                // response (a reasoning model can return only reasoning tokens or
                // whitespace) and to spot an announced-but-unperformed next step.
                let assistant_text: String = completion
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_text = !assistant_text.trim().is_empty();

                // Auto-recover from a content-less response — no tool calls and no
                // text, i.e. a flaky provider returning only reasoning or an empty
                // message. Silently re-run a few times before giving up, each
                // retry resampling hotter (see the temperature bump above). The
                // dead round isn't recorded, so each retry re-runs with the
                // original context.
                if calls.is_empty() && !has_text {
                    if empty_retries < self.config.max_empty_retries {
                        empty_retries += 1;
                        ui.status(&format!(
                            "⚠ the model returned no response — retrying ({empty_retries}/{})",
                            self.config.max_empty_retries
                        ));
                        continue;
                    }
                    ui.status(
                        "⚠ the model returned no response after retrying — try /retry, or \
                         /model to switch.",
                    );
                    break false;
                }
                // Real output this round — clear the retry counter so the
                // temperature bump is transient: a later, unrelated stall gets
                // its own budget rather than inheriting this one's elevation.
                empty_retries = 0;

                if calls.is_empty() {
                    // Text but no tool call (the content-less case was handled
                    // above): no tool_use blocks, so the safe text-only append
                    // applies. If the model was actively working this turn and its
                    // last line reads like an announced-but-unperformed next step
                    // ("Now let me rewrite main.rs:"), nudge it to actually do it —
                    // bounded — rather than ending the turn on a false "done".
                    self.messages.push_assistant(std::mem::take(&mut completion.content));
                    if made_tool_call
                        && continue_nudges < self.config.max_continue_nudges
                        && looks_like_unfinished_step(&assistant_text)
                    {
                        continue_nudges += 1;
                        stalled_unfinished = true;
                        ui.status(&format!(
                            "the model described a next step but didn't run it — \
                             nudging it to continue ({continue_nudges}/{})",
                            self.config.max_continue_nudges
                        ));
                        self.messages.push_nudge(NudgeKind::Continue, CONTINUE_NUDGE);
                        continue;
                    }
                    break false;
                }
                // The model requested tool calls — it's actively working, so a
                // later text-only stop is eligible for a nudge, and any pending
                // "incomplete" state is cleared (it got back to work).
                made_tool_call = true;
                stalled_unfinished = false;
                // Infer within-batch dependencies (a read of a file a mutating
                // call earlier in the batch targeted must observe that mutation;
                // mutating calls serialize). The scheduler below runs ready
                // calls concurrently respecting this graph, so independent reads
                // can overlap with an independent later write — while a read
                // whose path matches an earlier write waits for it.
                let deps = tool_deps(&calls);
                // Execute via a ready-queue scheduler over the dep graph. A call
                // is ready when all its deps are complete. Ready non-bash calls
                // run concurrently; bash runs alone this round (its line-by-line
                // UI streaming can't be reordered, and `tool_deps` already makes
                // it depend on all prior calls via the unknown-path fallback, so
                // it's never ready alongside a dependent). Results are collected
                // and recorded together via `push_assistant_with_results` so the
                // transcript never carries an orphan tool_use; results are
                // ordered by emission index so the transcript reads in model
                // order. UI streaming and snapshot invalidation still happen
                // during execution.
                let mut results: Vec<Option<(String, String)>> = vec![None; calls.len()];
                let mut completed = vec![false; calls.len()];
                let mut completion_order: Vec<usize> = Vec::with_capacity(calls.len());
                // Pre-pass: handle `record_decision` calls serially. They mutate
                // agent state (`self.decisions`) and aren't real tool dispatches,
                // so they can't run in the parallel `execute` stream (no `&mut
                // self` there). They're instantaneous and have no deps that
                // matter, so handling them up front is safe.
                for (i, (id, name, arguments)) in calls.iter().enumerate() {
                    if name != "record_decision" {
                        continue;
                    }
                    ui.tool_call(name, arguments);
                    let content = self.handle_record_decision(arguments);
                    ui.tool_result(&content);
                    results[i] = Some((id.clone(), content));
                    completed[i] = true;
                    completion_order.push(i);
                }
                let mut done = completion_order.len();
                // Proactive per-edit checks: kicked off in the background as
                // mutating calls complete, awaited after the batch so any
                // syntax/lint error surfaces during the turn (before turn-end
                // verify) while the edit is still the model's focus. Each entry
                // is (path, join handle of the check).
                let mut pending_checks: Vec<(
                    String,
                    tokio::task::JoinHandle<(bool, String)>,
                )> = Vec::new();
                while done < calls.len() {
                    // Ready: deps all complete.
                    let ready: Vec<usize> = (0..calls.len())
                        .filter(|&i| !completed[i] && deps[i].iter().all(|&d| completed[d]))
                        .collect();
                    if ready.is_empty() {
                        // Shouldn't happen (deps point backward) — break to
                        // avoid spinning if the graph were somehow cyclic.
                        break;
                    }
                    // If any ready call is bash, run it alone (streaming UI).
                    let bash_idx = ready
                        .iter()
                        .copied()
                        .find(|&i| calls[i].1 == "bash");
                    if let Some(i) = bash_idx {
                        let (id, name, arguments) = &calls[i];
                        ui.tool_call(name, arguments);
                        let ui_ref: &mut dyn Ui = &mut *ui;
                        let output = execute_streaming(name, arguments, &mut |line: &str| {
                            ui_ref.tool_result(line);
                        })
                        .await;
                        emit_tool_output(&mut *ui, &output);
                        results[i] = Some((id.clone(), output.content));
                        self.invalidate_snapshot();
                        completed[i] = true;
                        completion_order.push(i);
                        done += 1;
                        // Bash runs alone → a serial run and a batch of size 1.
                        sched_tool_calls += 1;
                        sched_serial_runs += 1;
                        sched_max_concurrent = sched_max_concurrent.max(1);
                        continue;
                    }
                    // Run all ready non-bash calls concurrently. Record the
                    // completion order as the ready order (within a concurrent
                    // batch, relative order doesn't matter — none depend on
                    // each other, or they wouldn't all be ready).
                    let batch_size = ready.len() as u32;
                    for &i in &ready {
                        ui.tool_call(&calls[i].1, &calls[i].2);
                    }
                    let outputs: Vec<_> = futures_util::stream::iter(
                        ready
                            .iter()
                            .map(|&i| execute(&calls[i].1, &calls[i].2)),
                    )
                    .buffered(self.config.max_parallel_tools)
                    .collect()
                    .await;
                    // Scheduler telemetry: this batch ran `batch_size` calls
                    // concurrently; a batch of 1 is a serial run.
                    sched_tool_calls += batch_size;
                    sched_max_concurrent = sched_max_concurrent.max(batch_size);
                    if batch_size == 1 {
                        sched_serial_runs += 1;
                    }
                    for (&i, output) in ready.iter().zip(outputs) {
                        emit_tool_output(&mut *ui, &output);
                        results[i] = Some((calls[i].0.clone(), output.content));
                        // Long-horizon: the model's `update_plan` statuses map
                        // onto the structured goal's sub-goals, so the agent
                        // advances/skips in lockstep with the model's stated
                        // progress. Only when long_horizon is on and a goal is
                        // set; the plan UI still renders via the ToolOutput.
                        if self.config.long_horizon
                            && calls[i].1 == "update_plan"
                            && let Some(goal) = self.structured_goal.as_mut()
                        {
                            apply_plan_to_goal(goal, &calls[i].2);
                        }
                        // A mutating tool may have changed files — invalidate
                        // the snapshot cache so a dependent read (guaranteed to
                        // run after by the dep graph) re-walks.
                        if !hi_tools::is_read_only(&calls[i].1) {
                            self.invalidate_snapshot();
                            // Proactive per-edit verify: kick off a background
                            // fast check for the edited file so a syntax/lint
                            // error surfaces during the turn. The check is
                            // awaited after the batch; failures are non-fatal.
                            if self.config.proactive_verify
                                && let Some(path) = hi_tools::target_path(&calls[i].1, &calls[i].2)
                                && let Some(cmd) = hi_tools::fast_check_for(&path)
                            {
                                let cmd = format!("{cmd} {path}");
                                pending_checks.push((
                                    path,
                                    tokio::spawn(async move { hi_tools::run_check(&cmd).await }),
                                ));
                            }
                        }
                        completed[i] = true;
                        completion_order.push(i);
                        done += 1;
                    }
                }
                // The completion order must respect the dep graph — a real
                // guarantee now (the scheduler only runs a call after its deps),
                // not just an emission-order coincidence.
                debug_assert!(
                    respects_deps(&deps, &completion_order),
                    "scheduler completion must respect inferred tool deps: {:?} vs {:?}",
                    deps,
                    completion_order
                );
                let results: Vec<(String, String)> = results.into_iter().flatten().collect();
                self.messages
                    .push_assistant_with_results(std::mem::take(&mut completion.content), results);
                // Await the proactive per-edit checks kicked off during the
                // batch and surface each as a status line — a syntax/lint error
                // appears here, during the turn, before turn-end verify. A pass
                // is silent (no need to noise a clean edit); a failure names the
                // file and shows the check output so the model can fix it now.
                for (path, handle) in pending_checks {
                    if let Ok((passed, output)) = handle.await {
                        if passed {
                            continue;
                        }
                        ui.status(&format!(
                            "⚠ proactive check failed for {path}:\n{output}"
                        ));
                    }
                }
            };

            if hit_cap {
                ui.status(&format!("reached step limit ({max_steps}); stopping turn"));
                ended_at_cap = true;
                break 'turn;
            }

            // Verification gate: run the stages in order (cheap compile/typecheck
            // first, then lint, then tests); the first to fail stops the turn and
            // its output is fed back. A passing pipeline ends the turn. The state
            // machine (round counter, change gating, stage execution) lives in the
            // `Verifier`; this loop just reacts to its outcome.
            let outcome = verifier
                .check(&turn_snapshot, &mut self.snapshot_cache, ui)
                .await;
            // Capture the verify snapshot for turn-end reuse whenever the
            // verifier actually walked the tree (i.e. it didn't bail before
            // snapshotting). On a failure we drop it: the model is about to edit
            // again, so it's no longer current.
            if matches!(outcome, VerifyOutcome::Passed | VerifyOutcome::Failed { .. }) {
                verify_snapshot = Some(self.snapshot_cached().await);
                if matches!(outcome, VerifyOutcome::Failed { .. }) {
                    verify_snapshot = None;
                }
            }
            match outcome {
                VerifyOutcome::NotRun => break 'turn,
                VerifyOutcome::SkippedNoChanges { first } => {
                    if first {
                        ui.status("verification skipped — no files changed this turn");
                    }
                    break 'turn;
                }
                VerifyOutcome::Passed => {
                    ui.status("✓ verification passed");
                    self.last_verify = Some(true);
                    break 'turn;
                }
                VerifyOutcome::Failed {
                    stage,
                    output,
                    round,
                } => {
                    ui.status(&format!("✗ {} failed; iterating", stage.name));
                    self.last_verify = Some(false);
                    let guidance = stage_guidance(&stage);
                    // Attribution: parse the (already-condensed) failure output
                    // into structured file/line/symbol hints and prepend a
                    // "Likely cause" section so the model is pointed at the
                    // right region first. Enrich-only — the raw `Output:` block
                    // stays unchanged, so nothing the model could see before is
                    // hidden. Empty when nothing parseable is found (the nudge
                    // then keeps its original shape).
                    let causes = hi_tools::parse_attributions(&output, 3);
                    // Capture for telemetry (flushed to the Agent at turn end).
                    last_verify_attributions = causes.clone();
                    let cause_section = if causes.is_empty() {
                        String::new()
                    } else {
                        let lines: Vec<String> = causes
                            .iter()
                            .map(|a| {
                                let kind = match a.kind {
                                    hi_tools::AttrKind::Compile => "compile",
                                    hi_tools::AttrKind::Test => "test",
                                    hi_tools::AttrKind::Lint => "lint",
                                    hi_tools::AttrKind::Other => "other",
                                };
                                let loc = match (a.line, a.column) {
                                    (Some(l), Some(c)) => format!("{}:{}:{}", a.path, l, c),
                                    (Some(l), None) => format!("{}:{}", a.path, l),
                                    _ => a.path.clone(),
                                };
                                if loc.is_empty() {
                                    format!("- [{kind}] {}", a.message)
                                } else {
                                    format!("- [{kind}] {loc} — {}", a.message)
                                }
                            })
                            .collect();
                        format!("Likely cause (verify and fix first):\n{}\n\n", lines.join("\n"))
                    };
                    let nudge_body = format!(
                        "{cause_section}Verification stage `{}` failed (`{}`).\n\nOutput:\n{}\n\n{} \
                         If a previous fix didn't work, reconsider rather than repeat it.",
                        stage.name, stage.command, output, guidance
                    );
                    // Replace the previous verify nudge instead of accumulating.
                    // Only the latest verification output belongs in context.
                    // `replace_last_nudge` pops trailing tool/assistant messages
                    // from the prior verify cycle and the prior nudge itself
                    // (located by typed kind, not string-matching), then pushes
                    // the new one. On the first round there's no prior nudge, so
                    // nothing is popped — the model's just-finished turn stays.
                    self.messages
                        .replace_last_nudge(NudgeKind::Verify { round }, nudge_body);
                }
            }
        }

        // Reuse the verify snapshot when available (verify passed or found no
        // changes — no model work happened since). Otherwise take a fresh one.
        let end_snapshot = match verify_snapshot.take() {
            Some(s) => s,
            None => self.snapshot_cached().await,
        };
        self.last_changed_files = changed_files_between(&turn_snapshot, &end_snapshot);
        self.last_compat_fallbacks = compat_fallbacks;
        // Flush the per-turn counters (otherwise discarded locals) into
        // telemetry so `--report` / the eval harness can diagnose the turn's
        // trajectory: how many verify rounds, recovery retries, nudges fired,
        // and where the last verify failure pointed.
        self.last_turn_telemetry = TurnTelemetry {
            verify_rounds: verifier.round(),
            recovery_retries: empty_retries,
            repeat_nudges,
            continue_nudges,
            hit_step_cap: ended_at_cap,
            stalled_unfinished,
            stalled_repeating,
            verify_attributions: last_verify_attributions
                .iter()
                .map(TurnAttribution::from)
                .collect(),
            tool_calls: sched_tool_calls,
            max_concurrent_batch: sched_max_concurrent,
            serial_runs: sched_serial_runs,
        };

        // Long-horizon driver: when a structured goal is set and long_horizon
        // is on, advance or retry the active sub-goal based on this turn's
        // outcome — so the next turn resumes coherently at the right sub-goal
        // (and with prior-attempt notes if it stalled). See `goal_turn_end`.
        self.goal_turn_end(stalled_unfinished, stalled_repeating, ended_at_cap, ui);

        // The model kept announcing steps it never ran, through the whole nudge
        // budget — don't let the turn read as a clean success.
        if stalled_unfinished {
            ui.status(
                "⚠ the model kept describing steps without running them — the task \
                 may be incomplete. /retry, or send 'continue'.",
            );
        }

        // Finalization: after a turn where the model used its tools to change
        // files, make one dedicated tool-free call so the user always gets a
        // structured recap, even from a model that wouldn't summarize on its
        // own. Requiring `made_tool_call` keeps a plain Q&A turn (whose answer is
        // already the response) from triggering it. Skipped when the turn
        // stalled or hit the step cap (the work may be incomplete, and a tidy
        // summary would misrepresent it) — the notices above stand instead.
        if self.config.finalize
            && made_tool_call
            && !ended_at_cap
            && !stalled_unfinished
            && !stalled_repeating
            && !self.last_changed_files.is_empty()
        {
            self.finalize_turn(turn_start, ui).await;
        }

        // Report cumulative session usage — the same number the live working
        // line and `/tokens` show, so the three never disagree.
        ui.turn_end(&self.usage_summary(&self.totals));
        self.persist()?;
        Ok(())
    }

    /// Make one dedicated, tool-free model call asking for a structured recap of
    /// the turn, and append it to the conversation as the closing assistant
    /// message. Best-effort: a provider error here doesn't fail the turn (the
    /// work is already done), it just leaves the turn without the extra summary.
    ///
    /// The synthetic request prompt is folded into history as a user turn so the
    /// roles stay alternating (some providers reject two assistant messages in a
    /// row) and the recap is part of the saved session.
    async fn finalize_turn(&mut self, turn_start: usize, ui: &mut dyn Ui) {
        // Only send the current turn's messages (plus the system prompt for
        // context), not the entire session history. The recap only needs to
        // know what happened *this turn* — sending 40K tokens of old context
        // to produce a 200-token summary is pure waste.
        let turn = &self.messages.as_slice()[turn_start..];
        let mut messages = Vec::with_capacity(turn.len() + 2);
        messages.push(self.minimal_system_message());
        messages.extend_from_slice(turn);
        messages.push(Message::user(FINALIZE_PROMPT));

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages: Arc::from(messages),
            tools: Arc::new([]), // recap only — no tool use
            max_tokens: 1024, // throwaway call — recaps are short
            temperature: self.config.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            profile: RequestProfile {
                compat: self.config.compat,
                tool_mode: ToolMode::ChatOnly,
                stream_usage: None,
            },
        };

        let mut recap = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                recap.push_str(&text);
                ui.assistant_text(&text);
            }
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Reasoning(_) => {}
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(completion) => completion,
            Err(err) => {
                self.add_error_usage(&err);
                ui.status(&format!("(couldn't generate the final summary: {err})"));
                return;
            }
        };
        ui.assistant_end();

        self.add_usage(completion.usage);
        ui.usage(
            self.totals.input_tokens,
            self.totals.output_tokens,
            self.context_used,
            self.config.context_window,
        );

        // Fall back to the final content if the provider didn't stream text.
        if recap.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(t) = c {
                    recap.push_str(t);
                }
            }
        }
        if recap.trim().is_empty() {
            return; // nothing to record
        }
        // Record both the synthetic request and the recap so roles alternate.
        // The recap is a text-only assistant message (no tool calls).
        self.messages.push_nudge(NudgeKind::Finalize, FINALIZE_PROMPT);
        self.messages.push_assistant(vec![Content::Text(recap)]);
    }

    /// Format a usage line. `usage` carries the cumulative in/out/total/cost;
    /// the context gauge instead uses `context_used` (the live conversation
    /// size), since cumulative input sums re-sent context across rounds and so
    /// isn't a measure of how full the window is.
    fn usage_summary(&self, usage: &hi_ai::Usage) -> String {
        // Cumulative session tokens, ↑ sent / ↓ received — these drive cost and
        // match the live working line. Abbreviated in the same units as the
        // context gauge below so the two never read as raw-vs-rounded.
        let mut summary = format!(
            "[↑{} ↓{}",
            humanize_count(usage.input_tokens),
            humanize_count(usage.output_tokens),
        );
        if usage.cache_read_tokens > 0 {
            summary.push_str(&format!(" ⟲{}", humanize_count(usage.cache_read_tokens)));
        }
        if let Some(cost) = self.cost_usd {
            summary.push_str(&format!(" · ${cost:.4}"));
        }
        // The context gauge is a *point-in-time* measure (the last request's
        // size), not cumulative input — so it is correctly smaller than ↑.
        if let Some(window) = self.config.context_window
            && window > 0
        {
            let pct = (self.context_used * 100 / u64::from(window)).min(100);
            summary.push_str(&format!(
                " · ctx {pct}% ({}/{})",
                humanize_count(self.context_used),
                humanize_count(u64::from(window)),
            ));
        }
        // Per-turn trajectory: a terse "steer" suffix when the turn needed
        // more than one shot, so a noisy success reads differently from a clean
        // one. Clean turns (no verify rounds, no recovery retries, no nudges,
        // no stalls) add nothing. See `TurnTelemetry`.
        if let Some(steer) = self.turn_steer() {
            summary.push_str(&format!(" · {steer}"));
        }
        summary.push(']');
        summary
    }

    /// A terse per-turn steering summary for the usage line, or `None` when the
    /// turn was clean (no extra rounds of any kind, no stall). Format:
    /// `steer: 2 verify · 1 retry · stalled` — components omitted when zero.
    fn turn_steer(&self) -> Option<String> {
        let t = &self.last_turn_telemetry;
        let mut parts: Vec<String> = Vec::new();
        if t.verify_rounds > 0 {
            parts.push(format!("{} verify", t.verify_rounds));
        }
        if t.recovery_retries > 0 {
            parts.push(format!("{} retry", t.recovery_retries));
        }
        if t.repeat_nudges > 0 {
            parts.push(format!("{} repeat", t.repeat_nudges));
        }
        if t.continue_nudges > 0 {
            parts.push(format!("{} continue", t.continue_nudges));
        }
        if t.stalled_unfinished || t.stalled_repeating {
            parts.push("stalled".to_string());
        }
        if parts.is_empty() {
            None
        } else {
            Some(format!("steer: {}", parts.join(" · ")))
        }
    }

    fn request_tools(&self) -> Arc<[ToolSpec]> {
        match self.config.tool_mode {
            ToolMode::ChatOnly => Arc::new([]),
            ToolMode::ReadOnly => self
                .tools
                .iter()
                .filter(|tool| hi_tools::is_read_only(&tool.name))
                .cloned()
                .collect::<Vec<_>>()
                .into(),
            ToolMode::Auto | ToolMode::Required => self.tools.clone().into(),
        }
    }

    fn tools_unavailable_for(&self, input: &str) -> bool {
        matches!(
            self.config.tool_mode,
            ToolMode::ChatOnly | ToolMode::ReadOnly
        ) && looks_mutating(input)
    }

    fn persist(&mut self) -> Result<()> {
        if let Some(session) = self.session.as_mut() {
            session.record(
                &self.messages.as_slice()[self.persisted..],
                self.totals,
                self.cost_usd,
            )?;
            self.persisted = self.messages.len();
        }
        Ok(())
    }

    /// Persist the current structured goal (if any) so a `/resume` picks it up
    /// at its active sub-goal. Best-effort: a failure is logged to the UI but
    /// doesn't fail the turn (the goal still lives in-memory for this session).
    fn persist_goal(&mut self, ui: &mut dyn Ui) {
        if let Some(session) = self.session.as_mut()
            && let Some(goal) = &self.structured_goal
        {
            if let Err(err) = session.record_goal(goal) {
                ui.status(&format!("(couldn't persist goal: {err})"));
            }
        }
    }

    /// Test-only direct access to the backing message vec, so tests can set up
    /// transcripts (prior turns, tool calls + results) without going through a
    /// model call. Goes through [`Transcript::mutate_slice`] so the same
    /// shared-`Arc` optimization applies.
    #[cfg(test)]
    pub(crate) fn messages_mut(&mut self) -> &mut Vec<Message> {
        self.messages.mutate_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hi_ai::{
        ChatRequest, Completion, Content, Provider, ProviderError, ProviderErrorKind, Role,
        StreamEvent, Usage,
    };
    use std::sync::{LazyLock, Mutex};

    /// A provider that returns canned completions in order.
    struct Canned(Mutex<Vec<Completion>>);

    #[async_trait]
    impl Provider for Canned {
        async fn stream(
            &self,
            _request: ChatRequest,
            _sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            Ok(self.0.lock().unwrap().remove(0))
        }
    }

    /// Like [`Canned`], but records each request's sampling tuple
    /// `(temperature, top_p, frequency_penalty)` (shared via an `Arc` so the test
    /// can inspect it after the provider is moved in).
    type Sample = (Option<f32>, Option<f32>, Option<f32>);
    struct RecordTemps {
        responses: Mutex<Vec<Completion>>,
        samples: std::sync::Arc<Mutex<Vec<Sample>>>,
    }

    #[async_trait]
    impl Provider for RecordTemps {
        async fn stream(
            &self,
            request: ChatRequest,
            _sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            self.samples.lock().unwrap().push((
                request.temperature,
                request.top_p,
                request.frequency_penalty,
            ));
            Ok(self.responses.lock().unwrap().remove(0))
        }
    }

    enum ProviderStep {
        Completion(Completion),
        RequestTooLarge,
        /// Fail this round with a provider error of the given kind.
        Error(ProviderErrorKind),
        ErrorWithUsage(ProviderErrorKind, Usage),
    }

    struct ScriptedProvider {
        steps: Mutex<Vec<ProviderStep>>,
        requests: std::sync::Arc<Mutex<Vec<Vec<Message>>>>,
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn stream(
            &self,
            request: ChatRequest,
            _sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            self.requests
                .lock()
                .unwrap()
                .push(request.messages.to_vec());
            match self.steps.lock().unwrap().remove(0) {
                ProviderStep::Completion(completion) => Ok(completion),
                ProviderStep::RequestTooLarge => Err(ProviderError::new(
                    ProviderErrorKind::RequestTooLarge,
                    "API error 400 Bad Request: chat input exceeds the maximum allowed size",
                )
                .into()),
                ProviderStep::Error(kind) => {
                    Err(ProviderError::new(kind, "scripted provider error").into())
                }
                ProviderStep::ErrorWithUsage(kind, usage) => {
                    Err(ProviderError::new(kind, "scripted provider error")
                        .with_usage(usage)
                        .into())
                }
            }
        }
    }

    struct NullUi;
    impl Ui for NullUi {
        fn assistant_text(&mut self, _: &str) {}
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str) {}
        fn status(&mut self, _: &str) {}
        fn turn_end(&mut self, _: &str) {}
    }

    type UsageRecords = std::sync::Arc<Mutex<Vec<(Usage, Option<f64>)>>>;

    struct RecordingSession {
        records: UsageRecords,
    }

    impl SessionSink for RecordingSession {
        fn record(
            &mut self,
            _messages: &[Message],
            usage: Usage,
            cost_usd: Option<f64>,
        ) -> Result<()> {
            self.records.lock().unwrap().push((usage, cost_usd));
            Ok(())
        }

        fn record_compaction(&mut self, _messages: &[Message]) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingUi {
        statuses: Vec<String>,
        turn_ends: Vec<String>,
    }

    impl Ui for RecordingUi {
        fn assistant_text(&mut self, _: &str) {}
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str) {}
        fn status(&mut self, s: &str) {
            self.statuses.push(s.to_string());
        }
        fn turn_end(&mut self, s: &str) {
            self.turn_ends.push(s.to_string());
        }
    }

    fn config() -> AgentConfig {
        AgentConfig {
            model: "m".into(),
            max_tokens: 100,
            max_verify_iterations: 2,
            auto_compact: false,
            // Default to summarize so the existing summarize/auto tests are
            // unaffected; hybrid/elide get dedicated tests.
            compaction: CompactionKind::Summarize,
            // Off by default so the canned-provider tests don't need an extra
            // completion for the recap; the finalization tests opt in.
            finalize: false,
            ..AgentConfig::default()
        }
    }

    fn completion(content: Vec<Content>, input: u64, output: u64) -> Completion {
        Completion {
            content,
            usage: Usage {
                input_tokens: input,
                output_tokens: output,
                ..Default::default()
            },
            stop_reason: None,
        }
    }

    fn agent(responses: Vec<Completion>, cfg: AgentConfig) -> Agent {
        Agent::new(Box::new(Canned(Mutex::new(responses))), cfg)
    }

    fn scripted_agent(
        steps: Vec<ProviderStep>,
        cfg: AgentConfig,
    ) -> (Agent, std::sync::Arc<Mutex<Vec<Vec<Message>>>>) {
        let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
        let provider = ScriptedProvider {
            steps: Mutex::new(steps),
            requests: requests.clone(),
        };
        (Agent::new(Box::new(provider), cfg), requests)
    }

    static VERIFY_TEST_LOCK: LazyLock<tokio::sync::Mutex<()>> =
        LazyLock::new(|| tokio::sync::Mutex::new(()));

    /// A completion that writes a throwaway file — marks the turn as having
    /// edited, so the (edit-gated) verification pipeline runs.
    fn write_completion(path: &str) -> Completion {
        completion(
            vec![Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: format!("{{\"path\":{path:?},\"content\":\"x\"}}"),
            }],
            1,
            1,
        )
    }

    /// A unique throwaway file path under the current workspace. The name is
    /// unique per *call* (not just per process), so concurrent test runs and
    /// repeated calls within one process never collide — and a file left
    /// behind by a test that panicked before cleanup doesn't get clobbered
    /// or mistaken for another test's artifact. The file lives in the
    /// workspace (cwd) on purpose: the verify snapshot walks `.` to detect
    /// changes, so the temp file must be inside it for verify to notice.
    fn temp_file(tag: &str) -> std::path::PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::current_dir()
            .unwrap()
            .join(format!(".hi-verify-{tag}-{}-{n}", std::process::id()))
    }

    #[tokio::test]
    async fn request_too_large_drops_prior_context_and_retries_latest_prompt() {
        let (mut agent, requests) = scripted_agent(
            vec![
                ProviderStep::RequestTooLarge,
                ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 12, 3)),
            ],
            config(),
        );
        let huge_old_output = "old tool output ".repeat(20_000);
        agent.messages_mut().push(Message::user("previous task"));
        agent.messages_mut().push(Message::assistant(vec![Content::ToolCall {
            id: "read-1".into(),
            name: "read".into(),
            arguments: r#"{"path":"LICENSE"}"#.into(),
        }]));
        agent.messages_mut()
            .push(Message::tool_result("read-1", huge_old_output.clone()));

        let mut ui = RecordingUi::default();
        agent
            .run_turn("fix the current bug", &mut ui)
            .await
            .unwrap();

        let requests = requests.lock().unwrap();
        let contains = |messages: &[Message], needle: &str| {
            messages.iter().flat_map(|m| &m.content).any(|c| match c {
                Content::Text(t) => t.contains(needle),
                Content::Thinking { text, .. } => text.contains(needle),
                Content::ToolCall {
                    name, arguments, ..
                } => name.contains(needle) || arguments.contains(needle),
                Content::ToolResult { output, .. } => output.contains(needle),
                _ => false,
            })
        };
        assert_eq!(requests.len(), 2);
        assert!(
            contains(&requests[0], &huge_old_output),
            "first request includes existing context"
        );
        assert!(
            !contains(&requests[1], &huge_old_output),
            "retry omits oversized prior context"
        );
        assert!(
            requests[1]
                .iter()
                .any(|m| m.text().contains("fix the current bug")),
            "latest user request is preserved"
        );
        assert!(
            ui.statuses
                .iter()
                .any(|s| s.contains("dropped prior conversation context")),
            "user sees recovery status: {:?}",
            ui.statuses
        );
        assert_eq!(agent.messages().last().unwrap().text(), "ok");
    }

    #[tokio::test]
    async fn request_too_large_latest_prompt_is_removed_after_failed_retry() {
        let (mut agent, _requests) = scripted_agent(vec![ProviderStep::RequestTooLarge], config());
        let start_len = agent.messages().len();
        let mut ui = RecordingUi::default();

        let err = agent
            .run_turn(&"single huge prompt ".repeat(20_000), &mut ui)
            .await
            .unwrap_err();

        assert_eq!(
            hi_ai::provider_error_kind(&err),
            Some(ProviderErrorKind::RequestTooLarge)
        );
        assert_eq!(
            agent.messages().len(),
            start_len,
            "failed oversized prompt is not left in live history"
        );
        assert!(
            ui.statuses.iter().any(|s| s.contains("shorten the prompt")),
            "user gets actionable status: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn malformed_stream_retries_and_recovers() {
        // A garbled stream on the first call is silently re-run (with recovery
        // sampling) rather than failing the turn — then it recovers.
        let (mut agent, requests) = scripted_agent(
            vec![
                ProviderStep::Error(ProviderErrorKind::MalformedStream),
                ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
            ],
            config(),
        );
        let mut ui = RecordingUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();

        assert_eq!(agent.messages().last().unwrap().text(), "recovered");
        assert_eq!(
            requests.lock().unwrap().len(),
            2,
            "retried once after the garble"
        );
        assert!(
            ui.statuses.iter().any(|s| s.contains("retrying")),
            "shows a retry, got: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn retry_counts_usage_from_failed_attempt() {
        let (mut agent, _requests) = scripted_agent(
            vec![
                ProviderStep::ErrorWithUsage(
                    ProviderErrorKind::MalformedStream,
                    Usage {
                        input_tokens: 7,
                        output_tokens: 100,
                        ..Default::default()
                    },
                ),
                ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
            ],
            config(),
        );

        agent.run_turn("go", &mut NullUi).await.unwrap();

        assert_eq!(agent.totals().input_tokens, 12);
        assert_eq!(agent.totals().output_tokens, 103);
    }

    #[tokio::test]
    async fn empty_completion_error_is_resampled_too() {
        // The same path catches a provider's empty-completion *error*, not just a
        // content-less Ok response.
        let (mut agent, requests) = scripted_agent(
            vec![
                ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
                ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
            ],
            config(),
        );
        agent.run_turn("go", &mut NullUi).await.unwrap();
        assert_eq!(agent.messages().last().unwrap().text(), "recovered");
        assert_eq!(requests.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn terminal_error_aborts_without_retry() {
        // A non-resamplable error (auth) fails the turn immediately — no retry.
        let (mut agent, requests) =
            scripted_agent(vec![ProviderStep::Error(ProviderErrorKind::Auth)], config());
        let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();
        assert_eq!(
            hi_ai::provider_error_kind(&err),
            Some(ProviderErrorKind::Auth)
        );
        assert_eq!(
            requests.lock().unwrap().len(),
            1,
            "a terminal error is not retried"
        );
    }

    #[tokio::test]
    async fn terminal_error_persists_usage_before_returning() {
        let records = std::sync::Arc::new(Mutex::new(Vec::new()));
        let (mut agent, _requests) = scripted_agent(
            vec![ProviderStep::ErrorWithUsage(
                ProviderErrorKind::Outage,
                Usage {
                    input_tokens: 11,
                    output_tokens: 100,
                    ..Default::default()
                },
            )],
            config(),
        );
        agent.set_session(Box::new(RecordingSession {
            records: records.clone(),
        }));

        let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

        assert_eq!(
            hi_ai::provider_error_kind(&err),
            Some(ProviderErrorKind::Outage)
        );
        assert_eq!(
            *records.lock().unwrap(),
            vec![(
                Usage {
                    input_tokens: 11,
                    output_tokens: 100,
                    ..Default::default()
                },
                None,
            )]
        );
    }

    #[tokio::test]
    async fn update_memory_writes_file_without_polluting_history() {
        let path = std::env::temp_dir().join(format!("hi-mem-{}-write.md", std::process::id()));
        let _ = std::fs::remove_file(&path);
        // The model returns a distilled bullet list.
        let mut agent = agent(
            vec![completion(
                vec![Content::Text(
                    "- always run cargo fmt\n- tests live in tests/".into(),
                )],
                7,
                4,
            )],
            config(),
        );
        let before = agent.messages().len();
        agent.update_memory_at(path.clone(), &mut NullUi).await;

        let written = std::fs::read_to_string(&path).expect("memory file written");
        let _ = std::fs::remove_file(&path);
        assert!(
            written.contains("always run cargo fmt"),
            "distilled: {written}"
        );
        assert_eq!(
            agent.messages().len(),
            before,
            "session history not polluted"
        );
        assert_eq!(agent.totals().output_tokens, 4, "usage counted");
    }

    #[tokio::test]
    async fn update_memory_persists_usage_without_new_messages() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-memory-persist-{}-{}.md",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let records = std::sync::Arc::new(Mutex::new(Vec::new()));
        let mut agent = agent(
            vec![completion(vec![Content::Text("- note".into())], 10, 5)],
            config(),
        );
        agent.set_session(Box::new(RecordingSession {
            records: records.clone(),
        }));

        agent.update_memory_at(path.clone(), &mut NullUi).await;
        let _ = std::fs::remove_file(path);

        assert_eq!(
            *records.lock().unwrap(),
            vec![(
                Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                None,
            )]
        );
    }

    #[tokio::test]
    async fn update_memory_is_best_effort_on_error() {
        // A provider error at quit must not panic or leave a file behind.
        let path = std::env::temp_dir().join(format!("hi-mem-{}-err.md", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let (mut agent, _requests) = scripted_agent(
            vec![ProviderStep::Error(ProviderErrorKind::Outage)],
            config(),
        );
        agent.update_memory_at(path.clone(), &mut NullUi).await;
        assert!(!path.exists(), "nothing written when distillation fails");
    }

    #[test]
    fn goal_updates_system_prompt_and_clear_history_keeps_it() {
        let mut agent = agent(vec![], config());
        agent.set_goal(Some("ship a stable TUI".into()));

        assert_eq!(agent.goal(), Some("ship a stable TUI"));
        assert!(
            agent.messages()[0]
                .text()
                .contains("[Current session goal]"),
            "goal marker included"
        );
        assert!(
            agent.messages()[0].text().contains("ship a stable TUI"),
            "goal text included"
        );

        agent.messages_mut().push(Message::user("noise"));
        agent.clear_history();
        assert_eq!(agent.messages().len(), 1);
        assert!(
            agent.messages()[0].text().contains("ship a stable TUI"),
            "goal survives clear-history"
        );

        agent.set_goal(None);
        assert_eq!(agent.goal(), None);
        assert!(
            !agent.messages()[0]
                .text()
                .contains("[Current session goal]"),
            "goal marker removed"
        );
    }

    #[tokio::test]
    async fn runs_a_tool_then_finishes() {
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "1".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo hi\"}".into(),
                }],
                5,
                1,
            ),
            completion(vec![Content::Text("all done".into())], 6, 2),
        ];
        let mut agent = agent(responses, config());
        agent.run_turn("do it", &mut NullUi).await.unwrap();

        let roles: Vec<Role> = agent.messages().iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![
                Role::System,
                Role::User,
                Role::Assistant, // tool call
                Role::Tool,      // tool result
                Role::Assistant, // final text
            ]
        );
        // Token totals accumulate across both model calls.
        assert_eq!(agent.totals().input_tokens, 11);
        assert_eq!(agent.totals().output_tokens, 3);
        assert_eq!(agent.messages().last().unwrap().text(), "all done");
    }

    #[tokio::test]
    async fn batched_read_only_tools_run_and_preserve_order() {
        // One round emits two read-only calls; both run (concurrently) and their
        // results are recorded back in call order. Reads resolve against the
        // crate dir (cargo sets cwd to the manifest dir).
        let responses = vec![
            completion(
                vec![
                    Content::ToolCall {
                        id: "1".into(),
                        name: "read".into(),
                        arguments: r#"{"path":"Cargo.toml"}"#.into(),
                    },
                    Content::ToolCall {
                        id: "2".into(),
                        name: "read".into(),
                        arguments: r#"{"path":"src/lib.rs"}"#.into(),
                    },
                ],
                5,
                1,
            ),
            completion(vec![Content::Text("done".into())], 6, 2),
        ];
        let mut agent = agent(responses, config());
        agent.run_turn("scan", &mut NullUi).await.unwrap();

        let outputs: Vec<String> = agent
            .messages()
            .iter()
            .flat_map(|m| &m.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 2, "both tool results recorded");
        assert!(
            outputs[0].contains("hi-agent"),
            "first result is Cargo.toml"
        );
        assert!(
            // The file's top-of-module doc comment — stable in the kept head even
            // after the per-result cap clips this (large) file's middle.
            outputs[1].contains("The agent loop"),
            "second result is lib.rs"
        );
    }

    #[tokio::test]
    async fn compact_replaces_history_with_summary() {
        let records = std::sync::Arc::new(Mutex::new(Vec::new()));
        let responses = vec![completion(
            vec![Content::Text(
                "BRIEF: ported the parser; tests green".into(),
            )],
            7,
            5,
        )];
        let mut agent = agent(responses, config());
        agent.set_session(Box::new(RecordingSession {
            records: records.clone(),
        }));
        // Some history to compact.
        agent.messages_mut().push(Message::user("hello"));
        agent.messages_mut()
            .push(Message::assistant(vec![Content::Text("hi".into())]));

        agent.compact(&mut NullUi).await.unwrap();

        // History collapses to system + summary.
        assert_eq!(agent.messages().len(), 2);
        assert_eq!(agent.messages()[0].role, Role::System);
        assert!(
            agent.messages()[1]
                .text()
                .contains("BRIEF: ported the parser"),
            "summary message retained"
        );
        // The summarization call's usage is counted.
        assert_eq!(agent.totals().output_tokens, 5);
        assert_eq!(
            *records.lock().unwrap(),
            vec![(
                Usage {
                    input_tokens: 7,
                    output_tokens: 5,
                    ..Default::default()
                },
                None,
            )],
            "manual compaction persists usage even though compacted messages are transient"
        );
    }

    #[tokio::test]
    async fn hybrid_keeps_recent_and_folds_summary() {
        let mut agent = agent(
            vec![completion(vec![Content::Text("OLD SUMMARY".into())], 3, 2)],
            config(),
        );
        // Two user turns; keep_recent = 1 summarizes the first, keeps the second.
        agent.messages_mut().push(Message::user("q1"));
        agent.messages_mut()
            .push(Message::assistant(vec![Content::Text("a1".into())]));
        agent.messages_mut().push(Message::user("q2"));
        agent.messages_mut()
            .push(Message::assistant(vec![Content::Text("a2".into())]));

        agent
            .compact_with(CompactionKind::Hybrid { keep_recent: 1 }, &mut NullUi)
            .await
            .unwrap();

        let m = agent.messages();
        // system + (summary folded into kept user turn) + kept assistant reply.
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].role, Role::System);
        assert_eq!(m[1].role, Role::User);
        assert!(
            m[1].text().contains("OLD SUMMARY"),
            "summary folded: {}",
            m[1].text()
        );
        assert!(
            m[1].text().contains("q2"),
            "recent turn kept: {}",
            m[1].text()
        );
        assert_eq!(m[2].text(), "a2");
        // No two consecutive same-role messages (provider-safe).
        assert!(
            m.windows(2).all(|w| w[0].role != w[1].role),
            "roles must alternate"
        );
    }

    #[tokio::test]
    async fn elide_then_summarize_tail_elides_tool_turns_summarizes_qa() {
        // A session with: an old tool-bearing turn (q1 + read + big result), an
        // old Q&A turn (q2 + text), and a recent turn (q3). The new default
        // strategy should elide the old tool result (keep the call/result
        // skeleton) and summarize only the old Q&A tail, folding the summary
        // into the first kept turn. The recent turn stays verbatim.
        let mut agent = agent(
            vec![completion(vec![Content::Text("QA SUMMARY".into())], 1, 1)],
            config(),
        );
        // Old tool-bearing turn.
        agent.messages_mut().push(Message::user("q1"));
        agent.messages_mut().push(Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
        agent
            .messages_mut()
            .push(Message::tool_result("c1", "x".repeat(500)));
        // Old Q&A turn (no tool results) — this is the conversational tail.
        agent.messages_mut().push(Message::user("q2"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text("a2".into())]));
        // Recent turn.
        agent.messages_mut().push(Message::user("q3"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text("a3".into())]));

        agent
            .compact_with(
                CompactionKind::ElideThenSummarizeTail { keep_recent: 1 },
                &mut NullUi,
            )
            .await
            .unwrap();

        let m = agent.messages();
        // The old tool result must be elided (skeleton kept, not wiped).
        let tool_results: Vec<&str> = m
            .iter()
            .flat_map(|msg| &msg.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            tool_results.iter().any(|o| o.starts_with("[elided")),
            "old tool result elided (skeleton kept): {tool_results:?}"
        );
        assert!(
            !tool_results.iter().any(|o| o.contains(&"x".repeat(100))),
            "old tool output content gone: {tool_results:?}"
        );
        // The Q&A summary is folded into the first kept turn (q3), and q3 stays.
        let user_texts: Vec<String> = m
            .iter()
            .filter(|msg| msg.role == Role::User)
            .map(|msg| msg.text())
            .collect();
        assert!(
            user_texts.iter().any(|t| t.contains("QA SUMMARY")),
            "Q&A tail summarized and folded: {user_texts:?}"
        );
        assert!(
            user_texts.iter().any(|t| t.contains("q3")),
            "recent turn kept: {user_texts:?}"
        );
        // Provider-safe: roles alternate.
        assert!(
            m.windows(2).all(|w| w[0].role != w[1].role),
            "roles must alternate: {:?}",
            m.iter().map(|x| x.role).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn elide_then_summarize_tail_skips_model_call_when_no_qa_tail() {
        // A pure tool-heavy session (no old Q&A turns): the strategy should
        // elide and NOT make a summarizing model call. Provide no canned
        // completion — if it tried to summarize, the provider would panic on
        // an empty response list.
        let mut agent = agent(vec![], config());
        agent.messages_mut().push(Message::user("q1"));
        agent.messages_mut().push(Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
        agent
            .messages_mut()
            .push(Message::tool_result("c1", "x".repeat(500)));
        agent.messages_mut().push(Message::user("q2"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text("a2".into())]));

        // keep_recent = 1 → q2 is recent; q1's tool result is old and gets
        // elided. No Q&A tail older than q2 → no model call.
        agent
            .compact_with(
                CompactionKind::ElideThenSummarizeTail { keep_recent: 1 },
                &mut NullUi,
            )
            .await
            .unwrap();
        let m = agent.messages();
        let tool_results: Vec<&str> = m
            .iter()
            .flat_map(|msg| &msg.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            tool_results.iter().any(|o| o.starts_with("[elided")),
            "old tool result elided: {tool_results:?}"
        );
    }

    #[tokio::test]
    async fn record_decision_persists_across_compaction_in_system_prompt() {
        // A decision recorded via the tool survives a compaction in the system
        // prompt (the log is injected into the system message, which compaction
        // preserves verbatim — not summarized away).
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "d1".into(),
                    name: "record_decision".into(),
                    arguments: r#"{"summary":"use BTreeMap","rationale":"ordered iteration","files":["src/m.rs"]}"#.into(),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
        ];
        let mut agent = agent(responses, config());
        agent.run_turn("refactor", &mut NullUi).await.unwrap();
        assert_eq!(agent.decisions().entries().len(), 1);
        assert_eq!(agent.decisions().entries()[0].summary, "use BTreeMap");

        // The system prompt contains the decision.
        let sys = agent.messages()[0].text();
        assert!(
            sys.contains("use BTreeMap") && sys.contains("ordered iteration"),
            "decision in system prompt: {sys}"
        );

        // A compaction that summarizes the Q&A tail must NOT remove the
        // decision from the system prompt.
        agent
            .compact_with(CompactionKind::Summarize, &mut NullUi)
            .await
            .unwrap();
        let sys_after = agent.messages()[0].text();
        assert!(
            sys_after.contains("use BTreeMap"),
            "decision survives compaction: {sys_after}"
        );
    }

    #[tokio::test]
    async fn proactive_verify_surfaces_a_per_edit_check_failure() {
        // With proactive_verify on, a write to a .py file with a syntax error
        // triggers a background `python3 -m py_compile` whose failure surfaces
        // as a status line during the turn (before turn-end verify). Skipped if
        // python3 isn't on PATH (the check just won't run).
        if std::process::Command::new("sh")
            .arg("-c")
            .arg("command -v python3")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: python3 not on PATH");
            return;
        }
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let mut cfg = config();
        cfg.proactive_verify = true;
        let tmp = temp_file("proactive");
        let py = tmp.with_extension("py");
        let p = py.to_string_lossy().to_string();
        // Write invalid Python so py_compile fails.
        let responses = vec![
            Completion {
                content: vec![Content::ToolCall {
                    id: "w".into(),
                    name: "write".into(),
                    arguments: format!(
                        r#"{{"path":{p:?},"content":"def (\n"}}"#
                    ),
                }],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    context_occupancy: 1,
                    ..Default::default()
                },
                stop_reason: None,
            },
            completion(vec![Content::Text("done".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("write it", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&py);
        // A proactive-check failure status line names the file.
        assert!(
            ui.statuses
                .iter()
                .any(|s| s.contains("proactive check failed") && s.contains(&p)),
            "proactive failure surfaced: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn structured_goal_state_injected_into_system_prompt_when_long_horizon_on() {
        // With long_horizon on, a structured goal's state (objective + sub-goal
        // checklist + retry notes) is injected into the system prompt so the
        // agent resumes the active sub-goal coherently each turn.
        let mut cfg = config();
        cfg.long_horizon = true;
        let mut agent = agent(vec![completion(vec![Content::Text("ok".into())], 1, 1)], cfg);
        let mut goal = Goal::new(
            "refactor the parser",
            vec!["write tests".into(), "rewrite parser".into()],
        );
        // Record a failed attempt so the prompt surfaces "don't repeat" notes.
        goal.record_failure("approach A didn't compile", DEFAULT_SUBGOAL_RETRIES);
        assert!(agent.set_structured_goal(Some(goal)), "accepted when long_horizon on");

        let sys = agent.messages()[0].text();
        assert!(sys.contains("Long-horizon goal"), "header: {sys}");
        assert!(sys.contains("refactor the parser"), "objective: {sys}");
        assert!(sys.contains("write tests"), "sub-goal: {sys}");
        assert!(
            sys.contains("don't repeat these"),
            "retry notes surfaced: {sys}"
        );

        // Clearing the goal removes the section.
        agent.set_structured_goal(None);
        let sys_after = agent.messages()[0].text();
        assert!(
            !sys_after.contains("Long-horizon goal"),
            "goal section cleared: {sys_after}"
        );
    }

    #[tokio::test]
    async fn structured_goal_rejected_when_long_horizon_off() {
        // Default config has long_horizon off — setting a structured goal is
        // rejected (the single-turn loop is unchanged), so the system prompt
        // gains no goal section.
        let mut agent = agent(vec![completion(vec![Content::Text("ok".into())], 1, 1)], config());
        let goal = Goal::new("do a thing", vec!["step one".into()]);
        assert!(!agent.set_structured_goal(Some(goal)), "rejected when off");
        assert!(agent.structured_goal().is_none());
        let sys = agent.messages()[0].text();
        assert!(!sys.contains("Long-horizon goal"), "no goal section when off: {sys}");
    }

    #[tokio::test]
    async fn long_horizon_driver_advances_on_clean_turn() {
        // With long_horizon on and a structured goal set, a turn that verifies
        // clean (or has no verify and doesn't stall) advances the active
        // sub-goal, and the system prompt reflects the new active sub-goal.
        let mut cfg = config();
        cfg.long_horizon = true;
        // One turn: model writes a file (tool), then a clean text finish. No
        // verify configured → a non-stalling turn with no verify is "clean".
        let tmp = temp_file("lh1");
        let p = tmp.to_string_lossy().to_string();
        let responses = vec![
            write_completion(&p),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        agent.set_structured_goal(Some(Goal::new(
            "refactor",
            vec!["step one".into(), "step two".into()],
        )));
        let mut ui = RecUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        let goal = agent.structured_goal().expect("goal still set");
        assert_eq!(goal.sub_goals[0].status, GoalStatus::Done, "advanced past step 1");
        assert_eq!(goal.active_index(), Some(1), "step 2 now active");
        // The system prompt reflects the new active sub-goal.
        assert!(
            agent.messages()[0].text().contains("step two"),
            "system prompt shows new active sub-goal"
        );
    }

    #[tokio::test]
    async fn long_horizon_driver_records_failure_on_stall() {
        // A turn that stalls (repeat guard exhausts) records a sub-goal attempt
        // so the next turn sees the prior note (and doesn't repeat the approach).
        let mut cfg = config();
        cfg.long_horizon = true;
        cfg.max_repeat_nudges = 1;
        // Model re-issues the same tool call → repeat guard stalls the turn
        // after exhausting the (1) nudge budget. Three identical writes: the
        // second triggers a nudge, the third exhausts the budget and breaks
        // stalled.
        let responses = vec![
            write_completion("lhstall"),
            write_completion("lhstall"),
            write_completion("lhstall"),
        ];
        let mut agent = agent(responses, cfg);
        agent.set_structured_goal(Some(Goal::new(
            "refactor",
            vec!["step one".into(), "step two".into()],
        )));
        let mut ui = RecUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();
        let _ = std::fs::remove_file("lhstall");
        let goal = agent.structured_goal().expect("goal still set");
        assert_eq!(goal.active_index(), Some(0), "didn't advance (stalled)");
        assert!(
            goal.sub_goals[0].attempts > 0,
            "recorded a failure attempt: {:?}",
            goal.sub_goals[0]
        );
        assert!(
            goal.sub_goals[0]
                .notes
                .iter()
                .any(|n| n.contains("stalled")),
            "stall reason recorded as a note: {:?}",
            goal.sub_goals[0].notes
        );
        // The system prompt surfaces the "don't repeat" notes on the active
        // sub-goal, so the next turn doesn't repeat the failed approach.
        assert!(
            agent.messages()[0]
                .text()
                .contains("don't repeat these"),
            "retry notes in system prompt"
        );
    }

    #[tokio::test]
    async fn scheduler_parallelism_counts_concurrent_batches() {
        // A batch of independent reads (different paths, no deps) should run
        // concurrently — telemetry reports max_concurrent_batch > 1 and a
        // sub-100% serial share. Pins that the dep-aware scheduler's
        // concurrency is measured, not just shipped on faith.
        let mut cfg = config();
        let responses = vec![completion(
            vec![
                Content::ToolCall {
                    id: "r1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"a.rs"}"#.into(),
                },
                Content::ToolCall {
                    id: "r2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"b.rs"}"#.into(),
                },
                Content::ToolCall {
                    id: "r3".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"c.rs"}"#.into(),
                },
            ],
            1,
            1,
        ),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1)];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("read them", &mut ui).await.unwrap();
        let tel = agent.last_turn_telemetry();
        assert_eq!(tel.tool_calls, 3, "three reads ran: {:?}", tel);
        assert!(
            tel.max_concurrent_batch >= 2,
            "independent reads overlapped: {:?}",
            tel
        );
        assert!(
            tel.serial_runs < tel.tool_calls,
            "not all serial: {:?}",
            tel
        );
    }

    #[tokio::test]
    async fn hybrid_falls_back_to_summarize_when_too_few_turns() {
        let mut agent = agent(
            vec![completion(
                vec![Content::Text("WHOLE SUMMARY".into())],
                1,
                1,
            )],
            config(),
        );
        agent.messages_mut().push(Message::user("only turn"));
        agent.messages_mut()
            .push(Message::assistant(vec![Content::Text("a".into())]));
        // keep_recent = 3 but only one turn → no recent window → summarize all.
        agent
            .compact_with(CompactionKind::Hybrid { keep_recent: 3 }, &mut NullUi)
            .await
            .unwrap();
        let m = agent.messages();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].role, Role::System);
        assert!(m[1].text().contains("WHOLE SUMMARY"));
    }

    #[tokio::test]
    async fn elide_shrinks_old_tool_output_without_a_model_call() {
        // Empty provider: if elision tried to call the model, this would panic.
        let mut agent = agent(vec![], config());
        let big = "x".repeat(500);
        agent.messages_mut().push(Message::user("read a"));
        agent.messages_mut().push(Message::assistant(vec![Content::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
        agent.messages_mut().push(Message::tool_result("c1", big.clone()));
        agent.messages_mut().push(Message::user("read b")); // recent turn
        agent.messages_mut().push(Message::assistant(vec![Content::ToolCall {
            id: "c2".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }]));
        agent.messages_mut().push(Message::tool_result("c2", big.clone()));

        agent
            .compact_with(
                CompactionKind::ElideToolOutput { keep_recent: 1 },
                &mut NullUi,
            )
            .await
            .unwrap();

        let outputs: Vec<String> = agent
            .messages()
            .iter()
            .flat_map(|m| &m.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect();
        assert!(
            outputs[0].starts_with("[elided"),
            "old elided: {}",
            outputs[0]
        );
        assert_eq!(outputs[1], big, "recent kept verbatim");
    }

    #[derive(Default)]
    struct RecUi {
        statuses: Vec<String>,
        usages: Vec<(u64, u64)>,
        turn_end: Option<String>,
        assistant: String,
    }
    impl Ui for RecUi {
        fn assistant_text(&mut self, t: &str) {
            self.assistant.push_str(t);
        }
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str) {}
        fn status(&mut self, t: &str) {
            self.statuses.push(t.to_string());
        }
        fn usage(
            &mut self,
            input_tokens: u64,
            output_tokens: u64,
            _ctx_used: u64,
            _ctx_win: Option<u32>,
        ) {
            self.usages.push((input_tokens, output_tokens));
        }
        fn turn_end(&mut self, summary: &str) {
            self.turn_end = Some(summary.to_string());
        }
    }

    /// A harmless tool-call round (runs `echo`), marking the turn as actively
    /// working so a later text-only stop is nudge-eligible.
    fn echo_call() -> Completion {
        completion(
            vec![Content::ToolCall {
                id: "t".into(),
                name: "bash".into(),
                arguments: "{\"command\":\"echo hi\"}".into(),
            }],
            1,
            1,
        )
    }

    #[tokio::test]
    async fn nudges_once_when_model_stalls_mid_step() {
        // Edited, then announced a next step without doing it, then — after the
        // nudge — actually did it and finished. One nudge, no incomplete notice.
        let responses = vec![
            echo_call(),
            completion(
                vec![Content::Text("Now let me rewrite main.rs:".into())],
                1,
                1,
            ),
            echo_call(),
            completion(vec![Content::Text("Done. Run cargo build.".into())], 1, 1),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("add a thing", &mut ui).await.unwrap();
        assert_eq!(
            ui.statuses.iter().filter(|s| s.contains("nudging")).count(),
            1,
            "exactly one nudge, got: {:?}",
            ui.statuses
        );
        assert!(
            !ui.statuses.iter().any(|s| s.contains("may be incomplete")),
            "no incomplete notice once it resumed, got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn completed");
    }

    #[tokio::test]
    async fn nudges_when_model_stalls_on_a_plan_list() {
        // Edited, then announced a multi-step plan as a numbered list without
        // doing it (the trailing line is a list item, which the old line-only
        // heuristic missed — the turn used to end here "without context"), then,
        // after the nudge, finished. One nudge, no incomplete notice.
        let responses = vec![
            echo_call(),
            completion(
                vec![Content::Text(
                    "Now let me make the fixes. Here's my plan:\n\n\
                     1. Remove unused deps\n2. Fix the gitignore duplicate"
                        .into(),
                )],
                1,
                1,
            ),
            echo_call(),
            completion(vec![Content::Text("Done. Run cargo test.".into())], 1, 1),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("clean it up", &mut ui).await.unwrap();
        assert_eq!(
            ui.statuses.iter().filter(|s| s.contains("nudging")).count(),
            1,
            "exactly one nudge, got: {:?}",
            ui.statuses
        );
        assert!(
            !ui.statuses.iter().any(|s| s.contains("may be incomplete")),
            "no incomplete notice once it resumed, got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn completed");
    }

    #[tokio::test]
    async fn gives_up_with_notice_after_cap() {
        // Worked once, then narrated a next step every round without doing it:
        // bounded to max_continue_nudges nudges, then an honest incomplete notice.
        let mut responses = vec![echo_call()];
        for _ in 0..(config().max_continue_nudges + 1) {
            responses.push(completion(
                vec![Content::Text("Now let me rewrite main.rs:".into())],
                1,
                1,
            ));
        }
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("add a thing", &mut ui).await.unwrap();
        assert_eq!(
            ui.statuses.iter().filter(|s| s.contains("nudging")).count(),
            config().max_continue_nudges as usize,
            "nudges are bounded, got: {:?}",
            ui.statuses
        );
        assert!(
            ui.statuses.iter().any(|s| s.contains("may be incomplete")),
            "incomplete notice after the cap, got: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn nudges_when_model_repeats_the_same_command() {
        // The model runs a command, then re-issues the *exact same* call next
        // round. The repetition guard nudges it to act on the output instead of
        // re-running, and the model then finishes. One repeat-nudge, no
        // "stuck repeating" notice.
        let responses = vec![
            echo_call(),
            echo_call(), // exact repeat → nudged
            completion(vec![Content::Text("Done. Run cargo test.".into())], 1, 1),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("check it", &mut ui).await.unwrap();
        assert_eq!(
            ui.statuses
                .iter()
                .filter(|s| s.contains("re-ran the same command"))
                .count(),
            1,
            "exactly one repeat-nudge, got: {:?}",
            ui.statuses
        );
        assert!(
            !ui.statuses.iter().any(|s| s.contains("kept re-running")),
            "no stuck-repeating notice once it moved on, got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn completed");
    }

    #[tokio::test]
    async fn gives_up_with_notice_after_repeat_cap() {
        // The model re-issues the exact same command every round, through the
        // whole repeat-nudge budget: bounded nudges, then an honest
        // "stuck repeating" notice.
        let mut responses = vec![echo_call()];
        for _ in 0..(config().max_repeat_nudges + 1) {
            responses.push(echo_call()); // exact repeat each round
        }
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("check it", &mut ui).await.unwrap();
        assert_eq!(
            ui.statuses
                .iter()
                .filter(|s| s.contains("re-ran the same command"))
                .count(),
            config().max_repeat_nudges as usize,
            "repeat-nudges are bounded, got: {:?}",
            ui.statuses
        );
        assert!(
            ui.statuses.iter().any(|s| s.contains("kept re-running")),
            "stuck-repeating notice after the cap, got: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn does_not_nudge_a_different_command() {
        // Two consecutive tool calls with different arguments are not a repeat —
        // both execute, no repeat-nudge.
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "t".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo one\"}".into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "t".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo two\"}".into(),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text("Done.".into())], 1, 1),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("run them", &mut ui).await.unwrap();
        assert!(
            !ui.statuses
                .iter()
                .any(|s| s.contains("re-ran the same command")),
            "different commands are not a repeat, got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn completed");
    }

    #[tokio::test]
    async fn does_not_nudge_a_plain_answer() {
        // No tool call this turn (a Q&A-style reply) — never nudge, never warn,
        // even though the text isn't an action.
        let responses = vec![completion(
            vec![Content::Text("The answer is 42.".into())],
            1,
            1,
        )];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("what is 6*7?", &mut ui).await.unwrap();
        assert!(
            !ui.statuses
                .iter()
                .any(|s| s.contains("nudging") || s.contains("incomplete")),
            "plain answer is left alone, got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn completed");
    }

    #[tokio::test]
    async fn finalizes_with_a_recap_when_files_changed() {
        // A turn that changes a file ends with a dedicated recap call, recorded
        // as the closing assistant message (preceded by the synthetic request so
        // roles alternate), with its usage counted.
        // Holds the workspace lock: this test writes a temp file, which would
        // otherwise perturb the file-change detection of the verify tests.
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let mut cfg = config();
        cfg.finalize = true;
        let tmp = temp_file("finalize");
        let p = tmp.to_string_lossy().to_string();
        let responses = vec![
            write_completion(&p),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(
                vec![Content::Text(
                    "## Summary\n- Created the file.\n\nRun `cargo test`.".into(),
                )],
                3,
                4,
            ),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("make a file", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&tmp);

        let m = agent.messages();
        assert_eq!(m.last().unwrap().role, Role::Assistant);
        assert!(
            m.last().unwrap().text().contains("## Summary"),
            "recap is the closing message: {}",
            m.last().unwrap().text()
        );
        assert_eq!(
            m[m.len() - 2].role,
            Role::User,
            "request precedes the recap"
        );
        // Roles alternate (no two assistants in a row → provider-safe next turn).
        assert!(
            m.windows(2).all(|w| w[0].role != w[1].role),
            "roles must alternate"
        );
        // The recap call's usage (3/4) is folded into the running totals.
        assert_eq!(agent.totals().input_tokens, 1 + 1 + 3);
        assert_eq!(agent.totals().output_tokens, 1 + 1 + 4);
    }

    #[tokio::test]
    async fn does_not_finalize_a_plain_answer() {
        // Finalization on, but the turn changed no files (a Q&A reply) — no extra
        // recap call fires. (The canned provider has exactly one completion; a
        // stray finalization call would panic trying to pop a second.)
        let mut cfg = config();
        cfg.finalize = true;
        let mut agent = agent(
            vec![completion(
                vec![Content::Text("The answer is 42.".into())],
                1,
                1,
            )],
            cfg,
        );
        let mut ui = RecUi::default();
        agent.run_turn("what is 6*7?", &mut ui).await.unwrap();
        let assistants = agent
            .messages()
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .count();
        assert_eq!(assistants, 1, "no extra recap message");
        assert_eq!(agent.totals().output_tokens, 1, "no extra recap call");
    }

    #[tokio::test]
    async fn turn_end_reports_cumulative_not_last_round() {
        // Two rounds (5/1 then 6/2). The done line must show the cumulative
        // session total (11/3/14), matching the live counter — not just the
        // last round (6/2/8).
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "1".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo hi\"}".into(),
                }],
                5,
                1,
            ),
            completion(vec![Content::Text("done".into())], 6, 2),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();
        let summary = ui.turn_end.expect("turn_end emitted");
        // Cumulative session totals (↑11 ↓3), matching the live counter — not just
        // the last round (↑6 ↓2).
        assert!(
            summary.contains("↑11 ↓3"),
            "cumulative totals, got: {summary}"
        );
    }

    #[tokio::test]
    async fn usage_line_separates_cumulative_spend_from_context_fill() {
        // The regression guard: with a window + price set, the done line shows
        // cumulative ↑/↓ session spend (abbreviated, matching the live line), the
        // cost, and a context gauge that is the *last request's* size — distinct
        // from cumulative input and humanized the same way. Pins against mixing
        // raw/abbreviated units, rendering a count two ways, or conflating the two.
        let mut cfg = config();
        cfg.context_window = Some(1_000_000);
        cfg.price = Some((5.0, 15.0)); // $/1M (in, out)
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "1".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo hi\"}".into(),
                }],
                8_000,
                100,
            ),
            completion(vec![Content::Text("done".into())], 12_000, 200),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();
        let line = ui.turn_end.expect("turn_end emitted");

        // Cumulative session spend, arrowed + abbreviated (same shape as the live line).
        assert!(line.contains("↑20k"), "cumulative input ↑ (8k+12k): {line}");
        assert!(
            line.contains("↓300"),
            "cumulative output ↓ (100+200): {line}"
        );
        // The context gauge is the LAST request (12k) over the window — NOT the
        // cumulative input (20k), and abbreviated, not raw.
        assert!(
            line.contains("ctx 1% (12k/1.0M)"),
            "point-in-time context: {line}"
        );
        // The old, mixed-unit, misleading format is gone.
        assert!(
            !line.contains(" in ·") && !line.contains("total"),
            "no raw in/out/total wording: {line}"
        );
        assert!(
            !line.contains("20000") && !line.contains("12000"),
            "no raw token counts: {line}"
        );
        // A clean turn (one tool call, no verify/retry/nudge) shows no steer
        // suffix — the trajectory surface is additive, only for noisy turns.
        assert!(
            !line.contains("steer"),
            "clean turn has no steer suffix: {line}"
        );
    }

    #[test]
    fn turn_steer_summarizes_trajectory() {
        // Clean turn → None.
        let mut a = agent(vec![], config());
        assert_eq!(a.turn_steer(), None);

        // Noisy turn → a steer line listing each non-zero component.
        a.last_turn_telemetry = TurnTelemetry {
            verify_rounds: 2,
            recovery_retries: 1,
            repeat_nudges: 0,
            continue_nudges: 0,
            hit_step_cap: false,
            stalled_unfinished: false,
            stalled_repeating: false,
            verify_attributions: Vec::new(),
            tool_calls: 0,
            max_concurrent_batch: 0,
            serial_runs: 0,
        };
        let steer = a.turn_steer().expect("noisy turn has a steer line");
        assert!(
            steer.contains("2 verify") && steer.contains("1 retry"),
            "lists non-zero components: {steer}"
        );
        assert!(
            !steer.contains("repeat") && !steer.contains("continue"),
            "omits zero components: {steer}"
        );

        // A stall is surfaced even with no rounds.
        a.last_turn_telemetry = TurnTelemetry {
            verify_rounds: 0,
            recovery_retries: 0,
            repeat_nudges: 0,
            continue_nudges: 0,
            hit_step_cap: false,
            stalled_unfinished: true,
            stalled_repeating: false,
            verify_attributions: Vec::new(),
            tool_calls: 0,
            max_concurrent_batch: 0,
            serial_runs: 0,
        };
        let steer = a.turn_steer().expect("stall has a steer line");
        assert!(steer.contains("stalled"), "stall flagged: {steer}");
    }

    #[tokio::test]
    async fn cost_accumulates_at_price_active_for_each_call() {
        let mut cfg = config();
        cfg.price = Some((1.0, 10.0));
        let responses = vec![
            completion(vec![Content::Text("first".into())], 1_000, 100),
            completion(vec![Content::Text("second".into())], 1_000, 100),
        ];
        let mut agent = agent(responses, cfg);

        agent.run_turn("first", &mut NullUi).await.unwrap();
        agent.set_model("m2".into(), Some((2.0, 20.0)), None);
        agent.run_turn("second", &mut NullUi).await.unwrap();

        assert_eq!(agent.cost_usd(), Some(0.006));
    }

    #[test]
    fn add_usage_uses_normalized_billable_across_provider_semantics() {
        // A session that switches providers mid-run must accrue cost coherently.
        // The `billable` breakdown is provider-computed, so the agent's cost
        // math doesn't have to know whether `input_tokens` includes cached
        // tokens (OpenAI) or excludes them (Anthropic). Pin: an OpenAI-style
        // usage where input_tokens already includes the cached subset must NOT
        // double-count the cached tokens, and an Anthropic-style usage where
        // input excludes cache must still bill the cache portion at a discount.
        let mut cfg = config();
        cfg.price = Some((1.0, 10.0)); // $/1M in, out
        let mut a = agent(vec![], cfg);

        // OpenAI-style: prompt_tokens=1000 includes 400 cached. The normalized
        // breakdown separates them: 600 regular + 400 cached. Cost must bill
        // 600 at full price + 400 at 0.5x — NOT 1000 + 400 (double-count).
        a.add_usage(Usage {
            input_tokens: 1000,
            output_tokens: 0,
            cache_read_tokens: 400,
            cache_creation_tokens: 0,
            input_includes_cache: true,
            context_occupancy: 1000,
            billable: Some(hi_ai::BillableBreakdown {
                regular_input: 600,
                cached_input: 400,
                cache_creation: 0,
                output: 0,
            }),
        });
        let openai_cost = a.cost_usd().unwrap();
        // 600*1 + 400*0.5 = 800 token-units -> $0.0008
        assert!(
            (openai_cost - 0.0008).abs() < 1e-9,
            "openai no double-count: {openai_cost}"
        );

        // Anthropic-style: input_tokens=600 excludes 400 cache_read + 100
        // cache_creation. The breakdown bills 600 regular + 400 at 0.5x + 100
        // at 1.25x. The agent must NOT re-derive (which would wrongly subtract
        // cache_read from input_tokens).
        a.add_usage(Usage {
            input_tokens: 600,
            output_tokens: 50,
            cache_read_tokens: 400,
            cache_creation_tokens: 100,
            input_includes_cache: false,
            context_occupancy: 1100,
            billable: Some(hi_ai::BillableBreakdown {
                regular_input: 600,
                cached_input: 400,
                cache_creation: 100,
                output: 50,
            }),
        });
        let total = a.cost_usd().unwrap();
        // anthropic increment: 600*1 + 400*0.5 + 100*1.25 + 50*10 = 600+200+125+500 = 1425 -> $0.001425
        assert!(
            (total - (0.0008 + 0.001425)).abs() < 1e-9,
            "coherent cumulative across providers: {total}"
        );
    }

    #[tokio::test]
    async fn emits_running_cumulative_usage_each_round() {
        // Two rounds (tool call, then text). The UI should see the cumulative
        // total climb after each round, so it can show usage live mid-turn.
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "1".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo hi\"}".into(),
                }],
                5,
                1,
            ),
            completion(vec![Content::Text("done".into())], 6, 2),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();
        // Cumulative after round 1 = (5,1); after round 2 = (11,3).
        assert_eq!(ui.usages, vec![(5, 1), (11, 3)]);
    }

    #[tokio::test]
    async fn auto_compacts_when_context_fills() {
        let mut cfg = config();
        cfg.auto_compact = true;
        cfg.context_window = Some(100);
        let responses = vec![
            completion(vec![Content::Text("ans1".into())], 90, 1), // fills context to 90%
            completion(vec![Content::Text("CONVO SUMMARY".into())], 5, 5), // the compaction call
            completion(vec![Content::Text("ans2".into())], 5, 1),  // turn two, post-compaction
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();

        agent.run_turn("q1", &mut ui).await.unwrap(); // starts empty → no compaction
        agent.run_turn("q2", &mut ui).await.unwrap(); // context 90% full → compacts first

        assert!(
            ui.statuses.iter().any(|s| s.contains("compacting")),
            "expected a compaction status, got {:?}",
            ui.statuses
        );
        assert!(
            agent
                .messages()
                .iter()
                .any(|m| m.text().contains("CONVO SUMMARY")),
            "history should be replaced by the summary"
        );
        assert_eq!(agent.messages().last().unwrap().text(), "ans2");
    }

    #[tokio::test]
    async fn elides_old_tool_outputs_before_model_request() {
        let mut cfg = config();
        cfg.auto_compact = true;
        cfg.context_window = Some(100);
        let (mut agent, requests) = scripted_agent(
            vec![ProviderStep::Completion(completion(
                vec![Content::Text("done".into())],
                5,
                1,
            ))],
            cfg,
        );
        agent.messages_mut().push(Message::user("existing long turn"));
        for i in 1..=8 {
            let id = format!("c{i}");
            agent.messages_mut().push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
            agent.messages_mut().push(Message::tool_result(
                &id,
                format!("{i}\n{}", "x".repeat(500)),
            ));
        }

        let mut ui = RecordingUi::default();
        agent.run_turn("continue", &mut ui).await.unwrap();

        let requests = requests.lock().unwrap();
        let outputs: Vec<String> = requests[0]
            .iter()
            .flat_map(|msg| &msg.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect();
        assert!(outputs[0].starts_with("[elided"), "{outputs:?}");
        assert!(outputs[1].starts_with("[elided"), "{outputs:?}");
        assert!(outputs[2].starts_with("3\n"), "{outputs:?}");
        assert!(outputs[7].starts_with("8\n"), "{outputs:?}");
        assert!(
            ui.statuses.iter().any(|s| s.contains("elided old tool")),
            "expected elision status, got {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn retry_uses_recovery_sampling() {
        // A content-less first round triggers the silent retry, which must
        // resample hotter and with nucleus + frequency penalty to escape the
        // attractor; the initial (non-retry) call uses the plain configured temp.
        let samples = std::sync::Arc::new(Mutex::new(Vec::new()));
        let provider = RecordTemps {
            responses: Mutex::new(vec![
                completion(vec![], 0, 0), // empty → retry
                completion(vec![Content::Text("recovered".into())], 5, 3),
            ]),
            samples: samples.clone(),
        };
        let mut cfg = config();
        cfg.temperature = Some(0.2);
        let mut agent = Agent::new(Box::new(provider), cfg);
        agent.run_turn("go", &mut NullUi).await.unwrap();

        let samples = samples.lock().unwrap();
        assert_eq!(
            samples.len(),
            2,
            "initial call + one retry, got {:?}",
            *samples
        );
        assert_eq!(
            samples[0],
            (Some(0.2), None, None),
            "first call: configured temp, no recovery overrides"
        );
        let (temp, top_p, freq) = samples[1];
        assert!(temp.unwrap() > 0.2, "retry resamples hotter, got {temp:?}");
        assert_eq!(top_p, Some(0.95), "retry adds nucleus sampling");
        assert!(
            freq.is_some_and(|f| f > 0.0),
            "retry adds a frequency penalty, got {freq:?}"
        );
    }

    #[tokio::test]
    async fn empty_response_recovers_on_retry() {
        // First round comes back content-less; the silent retry succeeds. The
        // dead round is dropped from history, so the retry sees the same context.
        let responses = vec![
            completion(vec![], 0, 0), // empty → retry
            completion(vec![Content::Text("here's the review".into())], 5, 3), // succeeds
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("review codebase", &mut ui).await.unwrap();
        assert!(
            ui.statuses.iter().any(|s| s.contains("retrying (1/")),
            "a retry should be shown, got: {:?}",
            ui.statuses
        );
        assert!(
            !ui.statuses.iter().any(|s| s.contains("after retrying")),
            "should not have given up, got: {:?}",
            ui.statuses
        );
        assert_eq!(agent.messages().last().unwrap().text(), "here's the review");
        // Only the successful assistant message is recorded (not the dead round).
        let assistants = agent
            .messages()
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .count();
        assert_eq!(assistants, 1);
    }

    #[tokio::test]
    async fn empty_response_gives_up_after_retries() {
        // Persistent content-less responses (the last is reasoning-only, which the
        // old zero-token check missed): exhaust the budget, then surface it.
        let responses = vec![
            completion(vec![], 0, 0),
            completion(vec![], 0, 0),
            completion(vec![], 0, 42),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("review codebase", &mut ui).await.unwrap();
        assert!(
            ui.statuses.iter().any(|s| s.contains("after retrying")),
            "exhaustion should be surfaced, got: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn normal_final_text_does_not_retry() {
        // A turn that ends with real text must not retry or warn.
        let mut agent = agent(
            vec![completion(
                vec![Content::Text("here's the review".into())],
                5,
                3,
            )],
            config(),
        );
        let mut ui = RecUi::default();
        agent.run_turn("review codebase", &mut ui).await.unwrap();
        assert!(
            !ui.statuses.iter().any(|s| s.contains("no response")),
            "real text should not warn, got: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn layered_verify_stops_at_first_failing_stage() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        // The compile gate fails, so the later (passing) test stage must NOT run
        // — and the feedback should be the compile-error guidance, not the test one.
        let mut cfg = config();
        cfg.verify = vec![
            VerifyStage::new("check", "false"), // "compile" fails
            VerifyStage::new("test", "true"),   // would pass, must be skipped
        ];
        cfg.max_verify_iterations = 1;
        // The model edits (so verification runs), then stops; after the failing
        // verify it re-prompts once more before the cap is reached.
        let tmp = temp_file("stop");
        let p = tmp.to_string_lossy().to_string();
        let mut agent = agent(
            vec![
                write_completion(&p),
                completion(vec![Content::Text("attempt 1".into())], 1, 1),
                completion(vec![Content::Text("attempt 2".into())], 1, 1),
            ],
            cfg,
        );
        let mut ui = RecUi::default();
        agent.run_turn("x", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(agent.last_verify(), Some(false));
        // The failing stage is named…
        assert!(
            ui.statuses
                .iter()
                .any(|s| s.contains("check") && s.contains("failed")),
            "names the failing stage: {:?}",
            ui.statuses
        );
        // …and the later test stage never ran (no status line for it).
        assert!(
            !ui.statuses.iter().any(|s| s.contains("· test:")),
            "test stage must be skipped after the gate fails: {:?}",
            ui.statuses
        );
        // …and the feedback to the model is the compile-error nudge.
        let fed_back = agent
            .messages()
            .iter()
            .any(|m| m.role == Role::User && m.text().contains("fix its root cause"));
        assert!(fed_back, "compile-stage guidance fed back");
        // The `false` command's output isn't a parseable diagnostic, so the
        // attribution layer adds no "Likely cause" section — the nudge keeps
        // its original shape (enrich-only contract).
        let has_cause = agent
            .messages()
            .iter()
            .any(|m| m.role == Role::User && m.text().contains("Likely cause"));
        assert!(
            !has_cause,
            "no attribution section for unparseable output"
        );
    }

    #[tokio::test]
    async fn layered_verify_passes_when_all_stages_pass() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let mut cfg = config();
        cfg.verify = vec![
            VerifyStage::new("check", "true"),
            VerifyStage::new("test", "true"),
        ];
        let tmp = temp_file("pass");
        let p = tmp.to_string_lossy().to_string();
        let mut agent = agent(
            vec![
                write_completion(&p),
                completion(vec![Content::Text("done".into())], 1, 1),
            ],
            cfg,
        );
        agent.run_turn("x", &mut NullUi).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(agent.last_verify(), Some(true));
    }

    #[tokio::test]
    async fn verify_failure_exhausts_retries() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let mut cfg = config();
        cfg.verify = vec![VerifyStage::new("test", "false")]; // always fails
        cfg.max_verify_iterations = 2;
        // The model edits once (so verify runs), then keeps finishing without
        // tool calls; verify fails each round until the cap.
        let tmp = temp_file("exhaust");
        let p = tmp.to_string_lossy().to_string();
        let responses = vec![
            write_completion(&p),
            completion(vec![Content::Text("attempt 1".into())], 1, 1),
            completion(vec![Content::Text("attempt 2".into())], 1, 1),
            completion(vec![Content::Text("attempt 3".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        agent.run_turn("x", &mut NullUi).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(agent.last_verify(), Some(false));
    }

    #[tokio::test]
    async fn verify_failure_nudge_carries_attribution() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        // A verify stage that emits a real rustc-style diagnostic should yield a
        // "Likely cause" section in the nudge pointing at the parsed file:line,
        // while the raw `Output:` block is preserved (enrich-only).
        let mut cfg = config();
        cfg.verify = vec![VerifyStage::new(
            "check",
            "printf 'error[E0308]: mismatched types\\n  --> src/lib.rs:42:18\\n' >&2; exit 1",
        )];
        cfg.max_verify_iterations = 1;
        let tmp = temp_file("attr");
        let p = tmp.to_string_lossy().to_string();
        let mut agent = agent(
            vec![
                write_completion(&p),
                completion(vec![Content::Text("attempt 1".into())], 1, 1),
                completion(vec![Content::Text("attempt 2".into())], 1, 1),
            ],
            cfg,
        );
        let mut ui = RecUi::default();
        agent.run_turn("x", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        // The attribution section is present and points at the parsed location.
        let nudge = agent
            .messages()
            .iter()
            .find(|m| m.role == Role::User && m.text().contains("Likely cause"))
            .expect("attribution section present");
        let body = nudge.text();
        assert!(
            body.contains("Likely cause (verify and fix first)"),
            "section header: {body}"
        );
        assert!(
            body.contains("src/lib.rs:42:18"),
            "parsed location in attribution: {body}"
        );
        assert!(
            body.contains("[compile]"),
            "compile kind label: {body}"
        );
        // Enrich-only: the raw output block is still there alongside it.
        assert!(
            body.contains("Output:\n"),
            "raw Output block preserved: {body}"
        );
        assert!(
            body.contains("mismatched types"),
            "raw error message preserved in Output block: {body}"
        );
    }

    #[tokio::test]
    async fn verify_skipped_when_no_files_changed() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        // A turn that only answers (no edits) must not run verification, even
        // when configured — so a red test suite can't hijack a question.
        let mut cfg = config();
        cfg.verify = vec![VerifyStage::new("test", "false")];
        let mut agent = agent(
            vec![completion(
                vec![Content::Text("just answering".into())],
                1,
                1,
            )],
            cfg,
        );
        let mut ui = RecUi::default();
        agent.run_turn("what does this do?", &mut ui).await.unwrap();
        assert_eq!(agent.last_verify(), None, "verify must not have run");
        assert!(
            ui.statuses
                .iter()
                .any(|s| s.contains("skipped — no files changed")),
            "skip is surfaced: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn verify_runs_when_bash_changes_files() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let tmp = temp_file("bash");
        let p = tmp.to_string_lossy().to_string();
        let mut cfg = config();
        cfg.verify = vec![VerifyStage::new("test", "true")];
        let mut agent = agent(
            vec![
                completion(
                    vec![Content::ToolCall {
                        id: "b".into(),
                        name: "bash".into(),
                        arguments: format!("{{\"command\":\"printf x > '{}'\"}}", p),
                    }],
                    1,
                    1,
                ),
                completion(vec![Content::Text("done".into())], 1, 1),
            ],
            cfg,
        );
        agent.run_turn("x", &mut NullUi).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(agent.last_verify(), Some(true));
    }
}
