//! The agent loop: user message → model → tool calls → results → repeat
//! until the model stops calling tools. No artificial step limit.

pub mod command;
pub mod compaction;
pub mod session;
pub mod ui;

use anyhow::Result;
use futures_util::StreamExt;
use hi_ai::{
    ChatRequest, CompatMode, Content, Message, Provider, RequestProfile, StreamEvent, ToolMode,
    ToolSpec, Usage,
};
use hi_tools::{execute, tool_specs};

pub use command::Command;
pub use compaction::{CompactionKind, DEFAULT_KEEP_RECENT};
pub use session::SessionSink;
pub use ui::{Ui, tool_label};

/// Auto-compact once the context window is at least this percent full.
const AUTO_COMPACT_PERCENT: u64 = 80;
/// After triggering, compact until the local estimate is at or below this
/// percent of the window (so there's headroom before the next compaction).
const COMPACT_TARGET_PERCENT: u64 = 50;
/// User turns auto-compaction keeps verbatim.
const AUTO_KEEP_RECENT: usize = 3;
/// How many times to silently re-run a round that came back with no tool calls
/// and no text (a flaky provider returning only reasoning or an empty message)
/// before giving up and surfacing it.
const MAX_EMPTY_RETRIES: u32 = 2;
/// Max read-only tool calls to run concurrently within one round, bounding the
/// open file handles / subprocesses a single batched response can spawn.
const MAX_PARALLEL_TOOLS: usize = 8;
/// Max times one turn will nudge a model that stopped after *describing* a next
/// step without performing it. Bounds the "narrate-then-stall" recovery;
/// `max_steps` is the hard backstop.
const MAX_CONTINUE_NUDGES: u32 = 2;
/// Sent when the model announces a next step but emits no tool call, to get it to
/// actually do the work instead of ending the turn on a false "done".
const CONTINUE_NUDGE: &str = "You described a next step but didn't run it. Continue now, \
using your tools, to actually make that change. If the task is genuinely already complete, \
stop and give your final recap: what you changed (by file) and the exact command to run or test it.";

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

/// Whether recovery sampling (a hotter resample on an empty/garbled retry) is on.
/// Off (`HI_RECOVERY_SAMPLING=0/off/false/no`) re-runs the retry at the configured
/// sampling — the knob for A/B-ing recovery on the eval harness. Read once.
static RECOVERY_SAMPLING: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    !matches!(
        std::env::var("HI_RECOVERY_SAMPLING").ok().as_deref(),
        Some("0" | "off" | "false" | "no")
    )
});

/// Sampling for a model round, escalating with the count of consecutive
/// content-less rounds (`retries`; 0 = the normal first attempt). Returns
/// `(temperature, top_p, frequency_penalty)`. On a normal round — or when recovery
/// sampling is disabled — it passes the configured temperature through and leaves
/// `top_p`/`frequency_penalty` at the provider default (`None`). On a retry it
/// climbs the temperature from a ≥0.7 floor toward 1.0 and adds nucleus sampling
/// plus a growing frequency penalty, so a repetition/garble loop is actively
/// discouraged rather than just reheated.
fn recovery_sampling(
    retries: u32,
    base_temperature: Option<f32>,
    enabled: bool,
) -> (Option<f32>, Option<f32>, Option<f32>) {
    if !enabled || retries == 0 {
        return (base_temperature, None, None);
    }
    let r = retries as f32;
    let temperature = (base_temperature.unwrap_or(0.7).max(0.7) + 0.15 * r).min(1.0);
    let frequency_penalty = (0.2 * r).min(0.6);
    (Some(temperature), Some(0.95), Some(frequency_penalty))
}

/// Instruction appended to a slice of history to summarize it for compaction.
const SUMMARIZE_PROMPT: &str = "Summarize our conversation so far into a concise but \
complete handoff brief: the task and goal, key decisions and constraints, files created \
or changed, commands that matter, and any open or next steps. This summary will REPLACE \
the history, so include everything needed to continue seamlessly. Output only the summary.";

const SYSTEM_PROMPT: &str = "\
You are hi, a coding agent running in the user's terminal, in their current \
working directory. You can read, write, and edit files and run shell commands \
via your tools. Work in the current directory and the existing project: if a \
build or package file (Cargo.toml, package.json, go.mod, pyproject.toml, …) is \
already present, modify it and its sources in place — do NOT scaffold a new \
nested sub-project or subdirectory for your work unless the user explicitly \
asks. Prefer making the change over describing it. Keep responses concise. \
For a non-trivial change, first take one line to state your plan (which files, \
what edits). Before finishing, re-read the regions you changed and verify they \
satisfy the request and are internally consistent — fix what you missed rather \
than declaring done prematurely.";

/// Ending instruction when no separate finalization step runs: the model itself
/// must produce the closing recap.
const SELF_RECAP_INSTRUCTION: &str = " When the task is done, stop and end with a short recap so \
the user has the full picture: a one-line headline of what you accomplished, then — for any \
non-trivial change — a brief bullet list of the key edits (grouped by file) and the exact \
command(s) to run or test it. Write it in past tense, covering only what you actually did; don't \
restate the plan or pad it. For a trivial change or a plain question, a single line is enough.";

/// Ending instruction when a finalization step will write the recap: the model
/// shouldn't duplicate it, just confirm completion.
const DEFERRED_RECAP_INSTRUCTION: &str = " When the task is done, stop. A separate step will write \
the final summary for the user, so you don't need to compose a full recap yourself — just make \
sure the work is actually complete and finish with at most a one-line note.";

/// The system message, optionally with project context and a session goal
/// appended. `finalize` selects the ending instruction: when a separate
/// finalization step will write the recap, the model is told not to duplicate it.
fn build_system(project_context: Option<&str>, goal: Option<&str>, finalize: bool) -> Message {
    let mut text = SYSTEM_PROMPT.to_string();
    text.push_str(if finalize {
        DEFERRED_RECAP_INSTRUCTION
    } else {
        SELF_RECAP_INSTRUCTION
    });
    // Ground the model in its real location so it doesn't guess paths (a wrong
    // `/home/user`, scaffolding under `/tmp`, copying from directories that don't
    // exist) and wander out of the project. Each shell command runs from here in
    // a fresh shell, so `cd` never persists — say so explicitly.
    if let Ok(cwd) = std::env::current_dir() {
        text.push_str(&format!(
            "\n\nYour working directory is `{}` — work here. Every shell command runs from \
             this directory in a fresh shell, so `cd` does NOT persist between commands. Use \
             paths relative to it; do not `cd` into, copy from, or create directories elsewhere.",
            cwd.display()
        ));
    }
    if let Some(context) = project_context
        && !context.trim().is_empty()
    {
        text.push_str("\n\n");
        text.push_str(context.trim());
    }
    if let Some(goal) = goal
        && !goal.trim().is_empty()
    {
        text.push_str("\n\n[Current session goal]\n");
        text.push_str(goal.trim());
    }
    Message::system(text)
}

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
    fn is_test(&self) -> bool {
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
}

pub struct Agent {
    provider: Box<dyn Provider>,
    config: AgentConfig,
    messages: Vec<Message>,
    tools: Vec<ToolSpec>,
    session: Option<Box<dyn SessionSink>>,
    /// How many messages have already been handed to the session sink.
    persisted: usize,
    /// Running total of tokens across the session.
    totals: Usage,
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
    /// Optional transient goal injected into the system prompt for future turns.
    goal: Option<String>,
}

impl Agent {
    /// Start a fresh session seeded with the system prompt.
    pub fn new(provider: Box<dyn Provider>, config: AgentConfig) -> Self {
        let system = build_system(config.project_context.as_deref(), None, config.finalize);
        Self::with_messages(provider, config, vec![system], 0)
    }

    /// Resume from previously-saved history (which already includes the system
    /// prompt). The loaded messages are treated as already persisted.
    pub fn resume(provider: Box<dyn Provider>, config: AgentConfig, history: Vec<Message>) -> Self {
        let persisted = history.len();
        Self::with_messages(provider, config, history, persisted)
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
            messages,
            tools: tool_specs(),
            session: None,
            persisted,
            totals: Usage::default(),
            last_verify: None,
            context_used: 0,
            checkpoints: Vec::new(),
            last_changed_files: Vec::new(),
            last_compat_fallbacks: Vec::new(),
            goal: None,
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
        Ok(Some(n))
    }

    /// Attach a sink that records messages produced from here on.
    pub fn set_session(&mut self, session: Box<dyn SessionSink>) {
        self.session = Some(session);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Discard messages back to `len` — used to drop an interrupted turn so the
    /// conversation stays consistent (no dangling user message).
    pub fn truncate_messages(&mut self, len: usize) {
        self.messages.truncate(len);
        self.persisted = self.persisted.min(self.messages.len());
    }

    /// Cumulative token usage across the session.
    pub fn totals(&self) -> &Usage {
        &self.totals
    }

    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

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
        self.messages = vec![self.system_message()];
        self.persisted = self.messages.len();
    }

    fn system_message(&self) -> Message {
        build_system(
            self.config.project_context.as_deref(),
            self.goal.as_deref(),
            self.config.finalize,
        )
    }

    fn refresh_system_message(&mut self) {
        let system = self.system_message();
        if let Some(first) = self.messages.first_mut() {
            *first = system;
        } else {
            self.messages.push(system);
        }
    }

    /// Current transient session goal, if any.
    pub fn goal(&self) -> Option<&str> {
        self.goal.as_deref()
    }

    /// Set or clear the transient session goal and inject it into the system prompt.
    pub fn set_goal(&mut self, goal: Option<String>) {
        self.goal = goal.and_then(|g| {
            let g = g.trim().to_string();
            (!g.is_empty()).then_some(g)
        });
        self.refresh_system_message();
    }

    /// Whether the most recent turn's verification passed (None if not run).
    pub fn last_verify(&self) -> Option<bool> {
        self.last_verify
    }

    pub fn last_changed_files(&self) -> &[String] {
        &self.last_changed_files
    }

    pub fn last_compat_fallbacks(&self) -> &[String] {
        &self.last_compat_fallbacks
    }

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
        }
    }

    /// Summarize the whole conversation and reset to system + summary.
    async fn compact_summarize(&mut self, ui: &mut dyn Ui) -> Result<()> {
        // Need at least one exchange beyond the system prompt to summarize.
        if self.messages.len() <= 1 {
            ui.status("nothing to compact yet");
            return Ok(());
        }
        // Own the slice so it doesn't borrow `self` across the `&mut self` call.
        let slice = self.messages[1..].to_vec();
        let Some(summary) = self.summarize(&slice, ui).await? else {
            ui.status("compaction produced no summary; keeping history");
            return Ok(());
        };
        let system = self.system_message();
        self.messages = vec![
            system,
            Message::user(format!("[Summary of the conversation so far]\n\n{summary}")),
        ];
        self.persisted = self.messages.len();
        ui.status("✓ compacted — context reset to the summary");
        Ok(())
    }

    /// Keep the last `keep_recent` user turns verbatim; summarize everything
    /// older and fold the brief into the first kept turn. Folding (rather than
    /// inserting a separate summary message) avoids two consecutive user
    /// messages, which some providers reject.
    async fn compact_hybrid(&mut self, keep_recent: usize, ui: &mut dyn Ui) -> Result<()> {
        let Some(split) = compaction::recent_split(&self.messages, keep_recent) else {
            // Nothing older than the recent window — summarize everything so a
            // triggered compaction still makes progress.
            return self.compact_summarize(ui).await;
        };
        let old = self.messages[1..split].to_vec();
        let Some(summary) = self.summarize(&old, ui).await? else {
            ui.status("compaction produced no summary; keeping history");
            return Ok(());
        };

        let system = self.system_message();
        let mut recent = self.messages[split..].to_vec();
        let head = recent[0].text();
        recent[0] = Message::user(format!(
            "[Summary of earlier conversation]\n\n{summary}\n\n---\n\n{head}"
        ));
        let mut next = Vec::with_capacity(recent.len() + 1);
        next.push(system);
        next.extend(recent);
        self.messages = next;
        self.persisted = self.messages.len();
        ui.status("✓ compacted — kept recent turns, summarized the rest");
        Ok(())
    }

    /// Deterministically shrink the bulky output of old tool calls. No model
    /// call. Mutates already-persisted messages in place; the session file keeps
    /// the originals, so this stays transient.
    fn compact_elide(&mut self, keep_recent: usize, ui: &mut dyn Ui) {
        // Only turns older than the recent window are eligible; if everything is
        // recent there's nothing to elide.
        let freed = match compaction::recent_split(&self.messages, keep_recent) {
            Some(split) => compaction::elide_tool_outputs(&mut self.messages, split),
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

    /// Run the summarization model call over `slice`, returning the summary text
    /// (trimmed), or `None` if the model produced nothing. Shared by the
    /// Summarize and Hybrid strategies.
    async fn summarize(&mut self, slice: &[Message], ui: &mut dyn Ui) -> Result<Option<String>> {
        ui.status("compacting the conversation…");

        let mut messages = Vec::with_capacity(slice.len() + 2);
        messages.push(self.system_message());
        messages.extend_from_slice(slice);
        messages.push(Message::user(SUMMARIZE_PROMPT));

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages,
            tools: Vec::new(), // summarizing — no tool use
            max_tokens: self.config.max_tokens,
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
        let completion = self.provider.stream(request, &mut sink).await?;
        ui.assistant_end();
        self.totals.input_tokens += completion.usage.input_tokens;
        self.totals.output_tokens += completion.usage.output_tokens;
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
            self.messages.push(Message::user(input));
            self.messages.push(Message::assistant(vec![Content::Text(format!(
                "I cannot perform coding actions in {} mode because file-edit and shell tools are unavailable. Switch to `--tool-mode auto` or `--tool-mode required` to let me modify the workspace.",
                tool_mode_label(self.config.tool_mode)
            ))]));
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
        if self.config.auto_compact
            && let Some(window) = self.config.context_window
            && window > 0
            && self.context_used * 100 >= u64::from(window) * AUTO_COMPACT_PERCENT
        {
            ui.status(&format!(
                "context ~{}% full — compacting to free room",
                self.context_used * 100 / u64::from(window)
            ));
            // Tier 1: deterministic, no model call. Only old turns are eligible.
            if let Some(split) = compaction::recent_split(&self.messages, AUTO_KEEP_RECENT) {
                compaction::elide_tool_outputs(&mut self.messages, split);
            }
            // Tier 2: only if still heavy. `context_used` reflects the
            // pre-elision request and is now stale, so gate on a local estimate.
            let target = u64::from(window) * COMPACT_TARGET_PERCENT / 100;
            if compaction::estimate_tokens(&self.messages) > target {
                let _ = self.compact(ui).await;
            }
            self.context_used = 0;
        }

        self.messages.push(Message::user(input));
        self.last_verify = None;
        self.last_changed_files.clear();
        self.last_compat_fallbacks.clear();
        let mut compat_fallbacks = Vec::new();

        let verify = self.config.verify.clone();
        let max_verify = self.config.max_verify_iterations;
        let max_steps = self.config.max_steps;
        let mut verify_round = 0u32;
        let mut steps = 0u32;
        let mut empty_retries = 0u32;
        let mut continue_nudges = 0u32;
        // Whether the model has run a tool this turn (so a later text-only stop is
        // eligible for a continue-nudge) and whether the turn ended on an
        // announced-but-unperformed step (drives the incomplete notice).
        let mut made_tool_call = false;
        let mut stalled_unfinished = false;
        // Whether the turn was cut short by the per-turn step cap, so the
        // finalization recap is skipped (the work may be incomplete).
        let mut ended_at_cap = false;
        // Snapshot the turn baseline so verification only runs when the
        // workspace ends up changed. This catches `bash` edits too, while
        // skipping verify when a turn makes no net file changes.
        let turn_snapshot = workspace_snapshot(std::path::Path::new("."));

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
                let (temperature, top_p, frequency_penalty) = recovery_sampling(
                    empty_retries,
                    self.config.temperature,
                    *RECOVERY_SAMPLING,
                );

                let request = ChatRequest {
                    model: self.config.model.clone(),
                    messages: self.messages.clone(),
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
                let completion = self.provider.stream(request, &mut sink).await?;
                ui.assistant_end();

                self.totals.input_tokens += completion.usage.input_tokens;
                self.totals.output_tokens += completion.usage.output_tokens;
                // Latest context fill, for the auto-compaction decision next turn.
                if completion.usage.input_tokens > 0 {
                    self.context_used = completion.usage.input_tokens;
                }
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
                    if empty_retries < MAX_EMPTY_RETRIES {
                        empty_retries += 1;
                        ui.status(&format!(
                            "⚠ the model returned no response — retrying ({empty_retries}/{MAX_EMPTY_RETRIES})"
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

                self.messages.push(Message::assistant(completion.content));
                if calls.is_empty() {
                    // Text but no tool call (the content-less case was handled
                    // above). If the model was actively working this turn and its
                    // last line reads like an announced-but-unperformed next step
                    // ("Now let me rewrite main.rs:"), nudge it to actually do it —
                    // bounded — rather than ending the turn on a false "done".
                    if made_tool_call
                        && continue_nudges < MAX_CONTINUE_NUDGES
                        && looks_like_unfinished_step(&assistant_text)
                    {
                        continue_nudges += 1;
                        stalled_unfinished = true;
                        ui.status(&format!(
                            "the model described a next step but didn't run it — \
                             nudging it to continue ({continue_nudges}/{MAX_CONTINUE_NUDGES})"
                        ));
                        self.messages
                            .push(Message::user(CONTINUE_NUDGE.to_string()));
                        continue;
                    }
                    break false;
                }
                // The model requested tool calls — it's actively working, so a
                // later text-only stop is eligible for a nudge, and any pending
                // "incomplete" state is cleared (it got back to work).
                made_tool_call = true;
                stalled_unfinished = false;
                // When the model batches several read-only calls (e.g. reading
                // many files to review them), run them concurrently — they have
                // no side effects and can't race. Any write/edit/bash in the
                // batch falls back to ordered, sequential execution so effects
                // stay isolated and in the order the model intended. Results are
                // recorded in call order either way.
                if calls.len() > 1
                    && calls
                        .iter()
                        .all(|(_, name, _)| hi_tools::is_read_only(name))
                {
                    for (_, name, arguments) in &calls {
                        ui.tool_call(name, arguments);
                    }
                    let outputs: Vec<_> = futures_util::stream::iter(
                        calls
                            .iter()
                            .map(|(_, name, arguments)| execute(name, arguments)),
                    )
                    .buffered(MAX_PARALLEL_TOOLS)
                    .collect()
                    .await;
                    for ((id, _, _), output) in calls.into_iter().zip(outputs) {
                        ui.tool_result(output.display.as_deref().unwrap_or(&output.content));
                        self.messages.push(Message::tool_result(id, output.content));
                    }
                } else {
                    for (id, name, arguments) in calls {
                        ui.tool_call(&name, &arguments);
                        let output = execute(&name, &arguments).await;
                        ui.tool_result(output.display.as_deref().unwrap_or(&output.content));
                        self.messages.push(Message::tool_result(id, output.content));
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
            // its output is fed back. A passing pipeline ends the turn.
            if verify.is_empty() || verify_round >= max_verify {
                break 'turn;
            }
            // Baseline-aware: only verify turns that changed files. A turn that
            // edited nothing can't have introduced a failure, so verifying would
            // only surface pre-existing/unrelated failures and pull the model
            // into fixing things the user didn't ask about.
            let changed_files = changed_files_between(
                &turn_snapshot,
                &workspace_snapshot(std::path::Path::new(".")),
            );
            if changed_files.is_empty() {
                if verify_round == 0 {
                    ui.status("verification skipped — no files changed this turn");
                }
                break 'turn;
            }
            verify_round += 1;
            let mut failure = None;
            for stage in &verify {
                ui.status(&format!(
                    "verifying ({verify_round}/{max_verify}) · {}: {}",
                    stage.name, stage.command
                ));
                let (passed, output) = hi_tools::run_check(&stage.command).await;
                if !passed {
                    failure = Some((stage, output));
                    break;
                }
            }
            match failure {
                None => {
                    ui.status("✓ verification passed");
                    self.last_verify = Some(true);
                    break 'turn;
                }
                Some((stage, output)) => {
                    ui.status(&format!("✗ {} failed; iterating", stage.name));
                    self.last_verify = Some(false);
                    // Compile/lint errors point at a cause; test failures imply a
                    // rule. Tailor the nudge so the model fixes the right thing.
                    let guidance = if stage.is_test() {
                        "These checks define the exact required behavior. Compare the expected \
                         and actual values to infer the precise rule — including edge cases and \
                         tie-breaking — then make the smallest edit that satisfies every case."
                    } else {
                        "Read the error above and fix its root cause (a type, name, or syntax \
                         problem) before anything else — the later stages can't run until this \
                         passes."
                    };
                    self.messages.push(Message::user(format!(
                        "Verification stage `{}` failed (`{}`).\n\nOutput:\n{}\n\n{} If a previous \
                         fix didn't work, reconsider rather than repeat it.",
                        stage.name, stage.command, output, guidance
                    )));
                }
            }
        }

        self.last_changed_files = changed_files_between(
            &turn_snapshot,
            &workspace_snapshot(std::path::Path::new(".")),
        );
        self.last_compat_fallbacks = compat_fallbacks;

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
            && !self.last_changed_files.is_empty()
        {
            self.finalize_turn(ui).await;
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
    async fn finalize_turn(&mut self, ui: &mut dyn Ui) {
        let mut messages = self.messages.clone();
        messages.push(Message::user(FINALIZE_PROMPT));

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages,
            tools: Vec::new(), // recap only — no tool use
            max_tokens: self.config.max_tokens,
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
                ui.status(&format!("(couldn't generate the final summary: {err})"));
                return;
            }
        };
        ui.assistant_end();

        self.totals.input_tokens += completion.usage.input_tokens;
        self.totals.output_tokens += completion.usage.output_tokens;
        if completion.usage.input_tokens > 0 {
            self.context_used = completion.usage.input_tokens;
        }
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
        self.messages.push(Message::user(FINALIZE_PROMPT));
        self.messages
            .push(Message::assistant(vec![Content::Text(recap)]));
    }

    /// Format a usage line. `usage` carries the cumulative in/out/total/cost;
    /// the context gauge instead uses `context_used` (the live conversation
    /// size), since cumulative input sums re-sent context across rounds and so
    /// isn't a measure of how full the window is.
    fn usage_summary(&self, usage: &hi_ai::Usage) -> String {
        let mut summary = format!(
            "[{} in · {} out · {} total",
            usage.input_tokens,
            usage.output_tokens,
            usage.total()
        );
        if let Some((input_price, output_price)) = self.config.price {
            let cost = (usage.input_tokens as f64 * input_price
                + usage.output_tokens as f64 * output_price)
                / 1_000_000.0;
            summary.push_str(&format!(" · ${cost:.4}"));
        }
        if let Some(window) = self.config.context_window
            && window > 0
        {
            let pct = (self.context_used * 100 / u64::from(window)).min(100);
            summary.push_str(&format!(
                " · {pct}% ctx ({}k/{}k)",
                self.context_used / 1000,
                window / 1000
            ));
        }
        summary.push(']');
        summary
    }

    fn request_tools(&self) -> Vec<ToolSpec> {
        match self.config.tool_mode {
            ToolMode::ChatOnly => Vec::new(),
            ToolMode::ReadOnly => self
                .tools
                .iter()
                .filter(|tool| hi_tools::is_read_only(&tool.name))
                .cloned()
                .collect(),
            ToolMode::Auto | ToolMode::Required => self.tools.clone(),
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
            session.record(&self.messages[self.persisted..])?;
            self.persisted = self.messages.len();
        }
        Ok(())
    }
}

fn tool_mode_label(mode: ToolMode) -> &'static str {
    match mode {
        ToolMode::Auto => "auto",
        ToolMode::Required => "required",
        ToolMode::ChatOnly => "chat-only",
        ToolMode::ReadOnly => "read-only",
    }
}

fn looks_mutating(input: &str) -> bool {
    let s = input.to_ascii_lowercase();
    [
        "edit",
        "fix",
        "change",
        "update",
        "write",
        "create",
        "delete",
        "remove",
        "rename",
        "implement",
        "add ",
        "modify",
        "refactor",
        "format",
        "run ",
    ]
    .iter()
    .any(|needle| s.contains(needle))
}

/// Heuristic: does the model's final text read like an *announced but unperformed*
/// next step — e.g. "Now let me rewrite main.rs:" or a "Here's my plan:" followed
/// by a numbered to-do list — rather than a finished answer or a past-tense recap?
///
/// It judges the trailing non-empty line, with one twist: when the message trails
/// off into a plan/to-do list, the intent lives in the line that *introduces* the
/// list ("Here's my plan:"), not the last bullet — so it judges that lead-in
/// instead, and only when the lead-in looks forward. That way a proper codex-style
/// recap that ends in a bullet list ("Key changes:\n- …") doesn't read as a stall,
/// while a model that announces a plan and quits without doing it does.
fn looks_like_unfinished_step(text: &str) -> bool {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let Some(&last) = lines.last() else {
        return false;
    };
    if is_list_item(last) {
        // Trailing plan/to-do list: unfinished only if the line introducing it
        // looks forward ("Here's my plan:"). A past-tense recap list is done.
        let lead = lines
            .iter()
            .rev()
            .find(|l| !is_list_item(l))
            .copied()
            .unwrap_or(last);
        return is_forward_intent(lead);
    }
    // Otherwise judge the trailing line: a dangling colon ("Now let me rewrite
    // main.rs:") or a forward-looking phrase means work was announced, not done.
    last.ends_with(':') || is_forward_intent(last)
}

/// Whether a line expresses *intent to act next* rather than a finished result.
fn is_forward_intent(line: &str) -> bool {
    let lower = line.to_lowercase();
    const FORWARD_INTENT: [&str; 12] = [
        "let me ",
        "let's ",
        "i'll ",
        "i will ",
        "i'm going to",
        "i am going to",
        "proceed to ",
        "here's my plan",
        "here is my plan",
        "my plan",
        "i need to ",
        "next, i",
    ];
    FORWARD_INTENT.iter().any(|phrase| lower.contains(phrase))
}

/// Whether a line is a markdown list item — a bullet (`- `, `* `, `• `) or a
/// numbered item (`1.`, `2)`) — used to spot a trailing plan/to-do list.
fn is_list_item(line: &str) -> bool {
    let l = line.trim_start();
    if l.starts_with("- ") || l.starts_with("* ") || l.starts_with("• ") {
        return true;
    }
    let digits = l.chars().take_while(|c| c.is_ascii_digit()).count();
    digits > 0 && l[digits..].starts_with(['.', ')'])
}

fn workspace_snapshot(dir: &std::path::Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    fn walk(
        base: &std::path::Path,
        dir: &std::path::Path,
        out: &mut std::collections::BTreeMap<String, Vec<u8>>,
    ) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if matches!(
                name.as_ref(),
                ".git" | "target" | "node_modules" | ".next" | "dist" | "build" | "__pycache__"
            ) {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                walk(base, &path, out);
            } else if let Ok(bytes) = std::fs::read(&path)
                && let Ok(rel) = path.strip_prefix(base)
            {
                out.insert(rel.to_string_lossy().into_owned(), bytes);
            }
        }
    }

    let mut out = std::collections::BTreeMap::new();
    walk(dir, dir, &mut out);
    out
}

fn changed_files_between(
    before: &std::collections::BTreeMap<String, Vec<u8>>,
    after: &std::collections::BTreeMap<String, Vec<u8>>,
) -> Vec<String> {
    let mut files = std::collections::BTreeSet::new();
    for path in before.keys() {
        if before.get(path) != after.get(path) {
            files.insert(path.clone());
        }
    }
    for path in after.keys() {
        if before.get(path) != after.get(path) {
            files.insert(path.clone());
        }
    }
    files.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hi_ai::{ChatRequest, Completion, Content, Provider, Role, StreamEvent, Usage};
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

    fn config() -> AgentConfig {
        AgentConfig {
            model: "m".into(),
            max_tokens: 100,
            temperature: None,
            thinking_budget: None,
            tool_mode: ToolMode::Auto,
            compat: CompatMode::Auto,
            price: None,
            context_window: None,
            project_context: None,
            verify: Vec::new(),
            max_verify_iterations: 2,
            max_steps: 50,
            auto_compact: false,
            // Default to summarize so the existing summarize/auto tests are
            // unaffected; hybrid/elide get dedicated tests.
            compaction: CompactionKind::Summarize,
            // Off by default so the canned-provider tests don't need an extra
            // completion for the recap; the finalization tests opt in.
            finalize: false,
        }
    }

    fn completion(content: Vec<Content>, input: u64, output: u64) -> Completion {
        Completion {
            content,
            usage: Usage {
                input_tokens: input,
                output_tokens: output,
            },
            stop_reason: None,
        }
    }

    fn agent(responses: Vec<Completion>, cfg: AgentConfig) -> Agent {
        Agent::new(Box::new(Canned(Mutex::new(responses))), cfg)
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

    /// A unique throwaway file path under the current workspace.
    fn temp_file(tag: &str) -> std::path::PathBuf {
        std::env::current_dir()
            .unwrap()
            .join(format!(".hi-verify-{tag}-{}", std::process::id()))
    }

    #[test]
    fn system_prompt_grounds_the_working_directory() {
        // The model must be told where it actually is, so it doesn't invent paths
        // (e.g. /home/user), cd elsewhere, or scaffold a new project.
        let sys = build_system(None, None, false);
        let text = sys.text();
        let cwd = std::env::current_dir().unwrap().display().to_string();
        assert!(text.contains(&cwd), "names the working directory: {text}");
        assert!(
            text.contains("does NOT persist"),
            "warns that cd doesn't persist"
        );
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

        agent.messages.push(Message::user("noise"));
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
        let responses = vec![completion(
            vec![Content::Text(
                "BRIEF: ported the parser; tests green".into(),
            )],
            7,
            5,
        )];
        let mut agent = agent(responses, config());
        // Some history to compact.
        agent.messages.push(Message::user("hello"));
        agent
            .messages
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
    }

    #[tokio::test]
    async fn hybrid_keeps_recent_and_folds_summary() {
        let mut agent = agent(
            vec![completion(vec![Content::Text("OLD SUMMARY".into())], 3, 2)],
            config(),
        );
        // Two user turns; keep_recent = 1 summarizes the first, keeps the second.
        agent.messages.push(Message::user("q1"));
        agent
            .messages
            .push(Message::assistant(vec![Content::Text("a1".into())]));
        agent.messages.push(Message::user("q2"));
        agent
            .messages
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
    async fn hybrid_falls_back_to_summarize_when_too_few_turns() {
        let mut agent = agent(
            vec![completion(
                vec![Content::Text("WHOLE SUMMARY".into())],
                1,
                1,
            )],
            config(),
        );
        agent.messages.push(Message::user("only turn"));
        agent
            .messages
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
        agent.messages.push(Message::user("read a"));
        agent
            .messages
            .push(Message::assistant(vec![Content::ToolCall {
                id: "c1".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
        agent.messages.push(Message::tool_result("c1", big.clone()));
        agent.messages.push(Message::user("read b")); // recent turn
        agent
            .messages
            .push(Message::assistant(vec![Content::ToolCall {
                id: "c2".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
        agent.messages.push(Message::tool_result("c2", big.clone()));

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
        // bounded to MAX_CONTINUE_NUDGES nudges, then an honest incomplete notice.
        let mut responses = vec![echo_call()];
        for _ in 0..(MAX_CONTINUE_NUDGES + 1) {
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
            MAX_CONTINUE_NUDGES as usize,
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
        assert_eq!(m[m.len() - 2].role, Role::User, "request precedes the recap");
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
            vec![completion(vec![Content::Text("The answer is 42.".into())], 1, 1)],
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

    #[test]
    fn unfinished_step_heuristic() {
        for t in [
            "Now let me rewrite main.rs:",
            "I'll add the struct",
            "Here is the plan:",
            // A "plan:" lead-in followed by a numbered to-do list — the trailing
            // line is a list item, so the lead-in is what's judged. (This is the
            // case the old line-only heuristic missed, ending the turn mid-plan.)
            "Now let me make the fixes. Here's my plan:\n\n1. Remove deps\n2. Fix gitignore\n3. Drop dead code",
        ] {
            assert!(looks_like_unfinished_step(t), "should flag: {t:?}");
        }
        for t in [
            "Done. Run `cargo build`.",
            "The answer is 42.",
            "I changed foo.rs and bar.rs.",
            // A past-tense recap that ends in a bullet list is finished, not a
            // stall — the lead-in ("Key changes:") looks back, not forward.
            "Key changes:\n- Added GOP support in encoder.rs\n- Updated the CLI in main.rs",
        ] {
            assert!(!looks_like_unfinished_step(t), "should not flag: {t:?}");
        }
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
        assert!(
            summary.contains("11 in · 3 out · 14 total"),
            "cumulative totals, got: {summary}"
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
        assert_eq!(samples.len(), 2, "initial call + one retry, got {:?}", *samples);
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

    #[test]
    fn recovery_sampling_escalates_and_toggles() {
        // Normal round: pass the configured temperature through, no overrides.
        assert_eq!(
            recovery_sampling(0, Some(0.2), true),
            (Some(0.2), None, None)
        );
        // First retry: hotter, with nucleus and a frequency penalty.
        let (t1, p1, f1) = recovery_sampling(1, Some(0.2), true);
        assert_eq!((p1, f1), (Some(0.95), Some(0.2)));
        assert!(t1.unwrap() > 0.2 && t1.unwrap() <= 1.0, "temp climbs: {t1:?}");
        // Second retry climbs further; temperature and penalty stay bounded.
        let (t2, _, f2) = recovery_sampling(2, Some(0.2), true);
        assert!(t2.unwrap() > t1.unwrap(), "temp keeps climbing");
        assert!(f2.unwrap() > f1.unwrap(), "penalty grows");
        assert!(t2.unwrap() <= 1.0 && f2.unwrap() <= 0.6, "both bounded");
        // Disabled: a retry behaves like a normal round (no overrides).
        assert_eq!(
            recovery_sampling(2, Some(0.2), false),
            (Some(0.2), None, None)
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
