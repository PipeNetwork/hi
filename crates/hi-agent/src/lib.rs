//! The agent loop: user message → model → tool calls → results → repeat
//! until the model stops calling tools. No artificial step limit.

pub mod command;
pub mod compaction;
pub mod session;
pub mod ui;

use anyhow::Result;
use futures_util::StreamExt;
use hi_ai::{ChatRequest, Content, Message, Provider, StreamEvent, ToolSpec, Usage};
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
than declaring done prematurely. When the task is done, stop, and end with one \
short line stating what you changed and the exact command to run or test it.";

/// The system message, optionally with project context and a session goal appended.
fn build_system(project_context: Option<&str>, goal: Option<&str>) -> Message {
    let mut text = SYSTEM_PROMPT.to_string();
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
    /// Optional transient goal injected into the system prompt for future turns.
    goal: Option<String>,
}

impl Agent {
    /// Start a fresh session seeded with the system prompt.
    pub fn new(provider: Box<dyn Provider>, config: AgentConfig) -> Self {
        let system = build_system(config.project_context.as_deref(), None);
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
        build_system(self.config.project_context.as_deref(), self.goal.as_deref())
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
            thinking_budget: None,
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

        let verify = self.config.verify.clone();
        let max_verify = self.config.max_verify_iterations;
        let max_steps = self.config.max_steps;
        let mut verify_round = 0u32;
        let mut steps = 0u32;
        let mut empty_retries = 0u32;

        'turn: loop {
            // Inner loop: model + tools until the model stops calling tools, or
            // the per-turn step cap is hit.
            let hit_cap = loop {
                if steps >= max_steps {
                    break true;
                }
                steps += 1;

                let request = ChatRequest {
                    model: self.config.model.clone(),
                    messages: self.messages.clone(),
                    tools: self.tools.clone(),
                    max_tokens: self.config.max_tokens,
                    temperature: self.config.temperature,
                    thinking_budget: self.config.thinking_budget,
                };

                let mut sink = |event: StreamEvent| match event {
                    StreamEvent::Text(text) => ui.assistant_text(&text),
                    StreamEvent::Reasoning(text) => ui.assistant_reasoning(&text),
                    StreamEvent::Status(text) => ui.status(&text),
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

                // Did this round produce any usable assistant text? (A reasoning
                // model can come back with only reasoning tokens, or whitespace,
                // and no real message.) Captured before the content is moved.
                let has_text = completion
                    .content
                    .iter()
                    .any(|c| matches!(c, Content::Text(t) if !t.trim().is_empty()));

                // Auto-recover from a content-less response — no tool calls and no
                // text, i.e. a flaky provider returning only reasoning or an empty
                // message. Silently re-run the same request a few times before
                // giving up. The dead round isn't recorded, so each retry re-runs
                // with the original context.
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

                self.messages.push(Message::assistant(completion.content));
                if calls.is_empty() {
                    break false;
                }
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
                break 'turn;
            }

            // Verification gate: run the stages in order (cheap compile/typecheck
            // first, then lint, then tests); the first to fail stops the turn and
            // its output is fed back. A passing pipeline ends the turn.
            if verify.is_empty() || verify_round >= max_verify {
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

        // Report cumulative session usage — the same number the live working
        // line and `/tokens` show, so the three never disagree.
        ui.turn_end(&self.usage_summary(&self.totals));
        self.persist()?;
        Ok(())
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

    fn persist(&mut self) -> Result<()> {
        if let Some(session) = self.session.as_mut() {
            session.record(&self.messages[self.persisted..])?;
            self.persisted = self.messages.len();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hi_ai::{ChatRequest, Completion, Content, Provider, Role, StreamEvent, Usage};
    use std::sync::Mutex;

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

    #[test]
    fn system_prompt_grounds_the_working_directory() {
        // The model must be told where it actually is, so it doesn't invent paths
        // (e.g. /home/user), cd elsewhere, or scaffold a new project.
        let sys = build_system(None, None);
        let text = sys.text();
        let cwd = std::env::current_dir().unwrap().display().to_string();
        assert!(text.contains(&cwd), "names the working directory: {text}");
        assert!(text.contains("does NOT persist"), "warns that cd doesn't persist");
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
            outputs[1].contains("fn run_turn"),
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
    }
    impl Ui for RecUi {
        fn assistant_text(&mut self, _: &str) {}
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
        // The compile gate fails, so the later (passing) test stage must NOT run
        // — and the feedback should be the compile-error guidance, not the test one.
        let mut cfg = config();
        cfg.verify = vec![
            VerifyStage::new("check", "false"), // "compile" fails
            VerifyStage::new("test", "true"),   // would pass, must be skipped
        ];
        cfg.max_verify_iterations = 1;
        // A turn re-prompts the model once after a failing verify, so two
        // completions are consumed (model → verify-fail → model → cap reached).
        let mut agent = agent(
            vec![
                completion(vec![Content::Text("attempt 1".into())], 1, 1),
                completion(vec![Content::Text("attempt 2".into())], 1, 1),
            ],
            cfg,
        );
        let mut ui = RecUi::default();
        agent.run_turn("x", &mut ui).await.unwrap();
        assert_eq!(agent.last_verify(), Some(false));
        // The failing stage is named…
        assert!(
            ui.statuses.iter().any(|s| s.contains("check") && s.contains("failed")),
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
        let mut cfg = config();
        cfg.verify = vec![VerifyStage::new("check", "true"), VerifyStage::new("test", "true")];
        let mut agent = agent(vec![completion(vec![Content::Text("done".into())], 1, 1)], cfg);
        agent.run_turn("x", &mut NullUi).await.unwrap();
        assert_eq!(agent.last_verify(), Some(true));
    }

    #[tokio::test]
    async fn verify_failure_exhausts_retries() {
        let mut cfg = config();
        cfg.verify = vec![VerifyStage::new("test", "false")]; // always fails
        cfg.max_verify_iterations = 2;
        // Each round the model "finishes" (no tool calls), so verify runs.
        let responses = vec![
            completion(vec![Content::Text("attempt 1".into())], 1, 1),
            completion(vec![Content::Text("attempt 2".into())], 1, 1),
            completion(vec![Content::Text("attempt 3".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        agent.run_turn("x", &mut NullUi).await.unwrap();
        assert_eq!(agent.last_verify(), Some(false));
    }
}
