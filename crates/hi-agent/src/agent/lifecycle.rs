//! `Agent` construction, accessors, and persistence: `new`/`resume`/`undo`,
//! the message/usage/context/goal/verify accessors, system-prompt refresh,
//! and `persist`/`persist_goal`/`messages_mut`.

use std::sync::Arc;

use anyhow::Result;
use hi_ai::{Message, Provider, Role, ToolMode, Usage, provider_error_usage};
use hi_tools::TOOL_SPECS;

use crate::compaction::{self, DEFAULT_KEEP_RECENT};
use crate::config::AgentConfig;
use crate::decision::DecisionLog;
use crate::goal::Goal;
use crate::heuristics::humanize_count;
use crate::prompt::SystemPrompt;
use crate::snapshot::SnapshotCache;
use crate::transcript::Transcript;
use crate::ui;
use crate::{SessionSink, TurnTelemetry, Ui, VerifyStage};

impl crate::Agent {
    /// Start a fresh session seeded with the system prompt.
    pub fn new(provider: Arc<dyn Provider>, config: AgentConfig) -> Self {
        let system = SystemPrompt::new()
            .with_project_context(config.project_context.as_deref())
            .with_finalize(config.finalize)
            .build();
        Self::with_messages(provider, config, vec![system], 0)
    }

    /// Resume from previously-saved history (which already includes the system
    /// prompt). The loaded messages are treated as already persisted.
    #[allow(clippy::too_many_arguments)]
    pub fn resume(
        provider: Arc<dyn Provider>,
        config: AgentConfig,
        history: Vec<Message>,
        usage: Usage,
        checkpoint_refs: Vec<String>,
        structured_goal: Option<Goal>,
        decisions: DecisionLog,
    ) -> Self {
        let persisted = history.len();
        let mut agent = Self::with_messages(provider, config, history, persisted);
        agent.totals = usage;
        agent.checkpoints = checkpoint_refs;
        if agent.checkpoints.len() > crate::MAX_CHECKPOINTS {
            agent
                .checkpoints
                .drain(0..agent.checkpoints.len() - crate::MAX_CHECKPOINTS);
        }
        agent.decisions = decisions;
        agent.structured_goal = agent
            .config
            .long_horizon
            .then_some(structured_goal)
            .flatten();
        agent.refresh_system_message();
        agent
    }

    fn with_messages(
        provider: Arc<dyn Provider>,
        config: AgentConfig,
        messages: Vec<Message>,
        persisted: usize,
    ) -> Self {
        let mut messages = Transcript::new(messages);
        // Clean up any stale synthetic nudges from a session saved by an older
        // version (before strip_finalize_pair existed). This prevents a resumed
        // session from carrying a FINALIZE_PROMPT ("don't take any further
        // action") into the next turn's context.
        messages.strip_finalize_pair();
        messages.strip_trailing_nudges();
        messages.repair_tool_result_ordering();
        messages.repair_invalid_tool_call_arguments();
        messages.repair_provider_invisible_assistant_messages();
        messages.repair_consecutive_assistant_messages();
        messages.repair_consecutive_user_messages();
        // Clamp persisted to the (possibly shorter) transcript length so the
        // incremental session recorder doesn't slice past the end.
        let persisted = persisted.min(messages.len());
        // Install the process-global LSP manager so the tool layer can reach
        // it. Synced to `config.lsp` at startup; `/lsp on|off` toggles at
        // runtime. The OnceLock means the first session's manager wins —
        // subsequent calls (e.g. resume) reuse the existing one.
        let lsp_root = std::env::current_dir().unwrap_or_else(|_| ".".into());
        let mgr = hi_lsp::LspManager::new(lsp_root);
        if config.lsp {
            // We can't `.await` here (not async), so spawn a blocking task to
            // flip the flag. The manager starts disabled by default.
            let mgr_arc = std::sync::Arc::new(mgr);
            hi_tools::set_lsp_manager_arc(mgr_arc.clone());
            // Fire-and-forget the enable — it'll be ready by the first query.
            tokio::spawn(async move {
                mgr_arc.set_enabled(true).await;
            });
        } else {
            hi_tools::set_lsp_manager(mgr);
        }
        let tools = advertised_tools(&config);
        Self {
            provider,
            config,
            messages,
            tools,
            session: None,
            delegate_runner: None,
            persisted,
            totals: Usage::default(),
            last_verify: None,
            context_used: 0,
            checkpoints: Vec::new(),
            last_changed_files: Vec::new(),
            auto_skills_written: 0,
            explore_subagents_used: 0,
            delegate_subagents_used: 0,
            last_compat_fallbacks: Vec::new(),
            interrupt: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            last_turn_telemetry: TurnTelemetry::default(),
            goal: None,
            structured_goal: None,
            decisions: DecisionLog::default(),
            snapshot_cache: SnapshotCache::default(),
            last_plan: Vec::new(),
        }
    }

    /// Revert the file changes the most recent turn made, restoring its git
    /// checkpoint. Returns `None` if there's nothing to undo, else the number of
    /// files restored or removed.
    pub async fn undo(&mut self) -> Result<Option<usize>> {
        let Some(target) = self.checkpoints.last().cloned() else {
            return Ok(None);
        };
        let n = hi_tools::checkpoint::restore(std::path::Path::new("."), &target).await?;
        let mut next = self.checkpoints.clone();
        next.pop();
        if let Some(session) = self.session.as_mut() {
            session.record_checkpoints(&next)?;
        }
        self.checkpoints = next;
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

    /// Attach the runner that executes write-capable `delegate` subagents. Without
    /// one, the `delegate` tool reports itself unavailable.
    pub fn set_delegate_runner(&mut self, runner: std::sync::Arc<dyn crate::DelegateRunner>) {
        self.delegate_runner = Some(runner);
    }

    /// Turn the write-capable `delegate` subagent on/off at runtime (the `/delegate`
    /// command) — re-advertises the tool set accordingly. A [`DelegateRunner`] must
    /// be attached for it to actually run.
    pub fn set_write_subagents(&mut self, on: bool) {
        self.config.write_subagents = on;
        self.tools = advertised_tools(&self.config);
    }

    /// Whether the `delegate` subagent is currently advertised.
    pub fn write_subagents_enabled(&self) -> bool {
        self.config.write_subagents
    }

    /// The current conversation history (including the system prompt).
    pub fn messages(&self) -> &[Message] {
        self.messages.as_slice()
    }

    /// The text of the last user message in the conversation, or `None` if
    /// there is none. Used by `/edit` to load it into the input line.
    pub fn last_user_message(&self) -> Option<String> {
        self.messages
            .as_slice()
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.text())
    }

    /// The text of the last assistant message, or `None`. Used to capture a
    /// read-only `explore` subagent's final answer after its turn completes.
    pub(crate) fn last_assistant_text(&self) -> Option<String> {
        self.messages
            .as_slice()
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .map(|m| m.text())
            .filter(|t| !t.trim().is_empty())
    }

    /// Discard messages back to `len` — used to drop an interrupted turn so the
    /// conversation stays consistent (no dangling user message, no orphan
    /// tool_use from a round cut off mid-execution).
    pub fn truncate_messages(&mut self, len: usize) {
        self.messages.rewind_to(len);
        self.persisted = self.persisted.min(self.messages.len());
    }

    /// Durably rewind the transcript to `len`. Used by explicit `/retry`, where
    /// the old attempt has already been persisted and must stay discarded after
    /// resume.
    pub fn truncate_messages_durable(&mut self, len: usize) -> Result<()> {
        let len = len.min(self.messages.len());
        let next = self.messages.as_slice()[..len].to_vec();
        self.replace_history_with_compaction(next)
    }

    /// Capture prompt-injected state before a turn starts, so `/retry` or an
    /// interrupt can discard the attempt without leaking decisions/goals/plans
    /// recorded during it.
    pub fn state_snapshot(&self) -> crate::AgentStateSnapshot {
        crate::AgentStateSnapshot {
            goal: self.goal.clone(),
            structured_goal: self.structured_goal.clone(),
            decisions: self.decisions.clone(),
            last_plan: self.last_plan.clone(),
        }
    }

    /// Live-only restore of a previously captured state snapshot. Used as a
    /// fallback after a failed durable discard so the current process still
    /// reflects the user's explicit interrupt.
    pub fn restore_state_snapshot(&mut self, snapshot: &crate::AgentStateSnapshot) {
        self.goal = snapshot.goal.clone();
        self.structured_goal = snapshot.structured_goal.clone();
        self.decisions = snapshot.decisions.clone();
        self.last_plan = snapshot.last_plan.clone();
        self.refresh_system_message();
    }

    /// Durably discard a turn by rewinding both the transcript and the
    /// prompt-injected side state to a pre-turn snapshot.
    pub fn rewind_to_snapshot_durable(
        &mut self,
        len: usize,
        snapshot: &crate::AgentStateSnapshot,
    ) -> Result<()> {
        let len = len.min(self.messages.len());
        let mut next = self.messages.as_slice()[..len].to_vec();
        let structured_goal = self
            .config
            .long_horizon
            .then_some(snapshot.structured_goal.clone())
            .flatten();
        let system = self.system_message_for(
            snapshot.goal.as_deref(),
            structured_goal.as_ref(),
            &snapshot.decisions,
        );
        if let Some(first) = next.first_mut() {
            *first = system;
        } else {
            next.push(system);
        }

        if let Some(session) = self.session.as_mut() {
            session.record_state_replacement(
                &next,
                structured_goal.as_ref(),
                &snapshot.decisions,
            )?;
        }
        self.messages.replace_all(next);
        self.persisted = self.messages.len();
        self.goal = snapshot.goal.clone();
        self.structured_goal = structured_goal;
        self.decisions = snapshot.decisions.clone();
        self.last_plan = snapshot.last_plan.clone();
        Ok(())
    }

    /// Cumulative token usage across the session.
    pub fn totals(&self) -> &Usage {
        &self.totals
    }

    /// The context-window occupancy, as last reported by the provider.
    pub fn context_used(&self) -> u64 {
        self.context_used
    }

    /// The configured context window, if known.
    pub fn context_window(&self) -> Option<u32> {
        self.config.context_window
    }

    /// Whether the LSP subsystem is enabled.
    pub fn lsp_enabled(&self) -> bool {
        self.config.lsp
    }

    /// Enable or disable the LSP subsystem at runtime (`/lsp on|off`).
    /// This updates the config flag and the process-global `LspManager`.
    pub fn set_lsp_enabled(&self, on: bool) {
        // The config field is behind a shared ref in some callers; we can't
        // mutate it directly. Instead, toggle the global manager, which is
        // what the tools actually check.
        // SAFETY: this is a single-threaded toggle from the REPL/TUI command
        // handler. The config.lsp field is only read at startup to seed the
        // manager; runtime checks go through the manager.
        if let Some(mgr) = hi_tools::lsp_manager_handle() {
            let mgr = mgr.clone();
            tokio::spawn(async move {
                mgr.set_enabled(on).await;
            });
        }
    }

    /// A human-readable context-occupancy breakdown for `/context`: the
    /// system prompt size, per-message token estimates, total occupancy vs.
    /// window, and what compaction would keep/elide.
    pub fn context_breakdown(&self) -> String {
        let messages = self.messages.as_slice();
        let window = self.config.context_window;
        let total_est = compaction::estimate_tokens(messages);
        let mut out = String::new();
        if let Some(w) = window
            && w > 0
        {
            let pct = (self.context_used * 100 / u64::from(w)).min(100);
            out.push_str(&format!(
                "context: {} / {} tokens ({}% used)\n",
                humanize_count(self.context_used),
                humanize_count(u64::from(w)),
                pct,
            ));
            out.push_str(&format!(
                "  estimated history: {} tokens\n",
                humanize_count(total_est),
            ));
            // How many turns until compaction triggers?
            let threshold = u64::from(w) * self.config.auto_compact_percent / 100;
            if self.context_used < threshold {
                let headroom = threshold.saturating_sub(self.context_used);
                out.push_str(&format!(
                    "  headroom before auto-compact: {} tokens ({})\n",
                    humanize_count(headroom),
                    if headroom > 0 {
                        "healthy"
                    } else {
                        "at threshold"
                    },
                ));
            } else {
                out.push_str(
                    "  ⚠ at or past the auto-compact threshold — /compact to reclaim now\n",
                );
            }
        } else {
            out.push_str(&format!(
                "context: {} tokens used (window unknown)\n",
                humanize_count(self.context_used),
            ));
        }
        // Per-message breakdown (system + up to 10 recent).
        out.push_str("\n  message breakdown:\n");
        for (i, msg) in messages.iter().enumerate().take(20) {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user  ",
                Role::Assistant => "asst  ",
                Role::Tool => "tool  ",
            };
            let est = compaction::estimate_tokens(std::slice::from_ref(msg));
            let preview = ui::clip(&msg.text().replace('\n', " "), 50);
            out.push_str(&format!(
                "    {i:>3} {role} ~{:<6} {preview}\n",
                humanize_count(est)
            ));
        }
        if messages.len() > 20 {
            out.push_str(&format!("    … {} more messages\n", messages.len() - 20));
        }
        // Compaction preview.
        out.push_str(&format!(
            "\n  compaction strategy: {:?}\n",
            self.config.compaction
        ));
        if let Some(split) = compaction::recent_split(messages, DEFAULT_KEEP_RECENT) {
            let old = split - 1;
            let recent = messages.len() - split;
            out.push_str(&format!(
                "  on compact: summarize {old} old, keep {recent} recent verbatim\n",
            ));
        } else {
            out.push_str("  on compact: nothing older than the recent window to summarize\n");
        }
        out
    }

    /// Render the conversation as Markdown for /export.
    pub fn export_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "# hi session transcript

",
        );
        for msg in self.messages.as_slice().iter() {
            match msg.role {
                hi_ai::Role::System => {} // skip system prompt
                hi_ai::Role::User => {
                    out.push_str(
                        "**user:**

",
                    );
                    out.push_str(&msg.text());
                    out.push_str(
                        "

",
                    );
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

    pub(crate) fn add_usage(&mut self, usage: Usage) {
        self.totals.add(usage);
        let effective_input = usage.effective_input_tokens();
        if effective_input > 0 {
            self.context_used = effective_input;
        }
    }

    pub(crate) fn add_error_usage(&mut self, err: &anyhow::Error) {
        self.add_usage(provider_error_usage(err));
    }

    pub(crate) fn emit_usage(&self, ui: &mut dyn Ui) {
        ui.usage(
            self.totals.input_tokens,
            self.totals.output_tokens,
            self.context_used,
            self.config.context_window,
        );
        ui.rate_limits(self.totals.rate_limits);
    }

    /// Number of git checkpoints created so far (for `/undo`).
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// A shared interrupt handle the UI can set to skip the current tool call.
    /// The agent checks it between tool executions; when set, the current tool's
    /// result is replaced with "interrupted by user" and the flag is cleared.
    pub fn interrupt_handle(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.interrupt.clone()
    }

    /// The model id currently configured for this session.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Capture the model and token/window settings so a caller can temporarily
    /// use a different model for one turn and restore the previous route exactly.
    pub fn model_state(&self) -> crate::AgentModelState {
        crate::AgentModelState {
            model: self.config.model.clone(),
            context_window: self.config.context_window,
            requested_max_tokens: self.config.requested_max_tokens,
            max_tokens: self.config.max_tokens,
            max_tokens_explicit: self.config.max_tokens_explicit,
        }
    }

    /// Restore a model state captured by [`Agent::model_state`].
    pub fn restore_model_state(&mut self, state: crate::AgentModelState) {
        self.config.model = state.model;
        self.config.context_window = state.context_window;
        self.config.requested_max_tokens = state.requested_max_tokens;
        self.config.max_tokens = state.max_tokens;
        self.config.max_tokens_explicit = state.max_tokens_explicit;
    }

    /// Switch the model used for subsequent turns, refreshing live metadata
    /// that drives the usage display and output-token budget.
    pub fn set_model(
        &mut self,
        model: String,
        context_window: Option<u32>,
        max_output_tokens: Option<u32>,
    ) {
        self.config.model = model;
        self.config.context_window = context_window;
        self.config.max_tokens = hi_ai::effective_coding_agent_max_tokens(
            &self.config.model,
            self.config.requested_max_tokens,
            self.config.max_tokens_explicit,
            max_output_tokens,
        );
    }

    /// Update the provider (endpoint + wire format + key) and model for subsequent
    /// turns. Used by `/provider` to use profiles mid-session. The caller
    /// builds the new `Arc<dyn Provider>` (e.g. Anthropic vs OpenAI adapter) and
    /// supplies a model id; pricing/context metadata is refreshed from the
    /// registry or the provider's live `/models` response.
    ///
    /// Safe to call only between turns (the REPL/TUI serialize turns, so a
    /// command handler runs when no stream is in flight). The conversation
    /// history is kept — the new provider sees the same messages, just routed to
    /// a different endpoint.
    pub fn set_provider(
        &mut self,
        provider: Arc<dyn Provider>,
        model: String,
        context_window: Option<u32>,
        requested_max_tokens: u32,
        max_tokens_explicit: bool,
        max_output_tokens: Option<u32>,
    ) {
        self.provider = provider;
        self.config.requested_max_tokens = requested_max_tokens;
        self.config.max_tokens_explicit = max_tokens_explicit;
        self.set_model(model, context_window, max_output_tokens);
    }

    /// Reset the live and persisted context to just the current system prompt.
    pub fn clear_history(&mut self) -> Result<()> {
        self.replace_history_with_compaction(vec![self.system_message()])
    }

    pub(crate) fn replace_history_with_compaction(&mut self, messages: Vec<Message>) -> Result<()> {
        if let Some(session) = self.session.as_mut() {
            session.record_compaction(&messages)?;
        }
        self.messages.replace_all(messages);
        self.persisted = self.messages.len();
        Ok(())
    }

    fn system_message_for(
        &self,
        goal: Option<&str>,
        structured_goal: Option<&Goal>,
        decisions: &DecisionLog,
    ) -> Message {
        let goal_section = structured_goal.and_then(|g| g.prompt_section());
        SystemPrompt::new()
            .with_project_context(self.config.project_context.as_deref())
            .with_goal(goal)
            .with_goal_state(goal_section.as_deref())
            .with_decisions(decisions.prompt_section().as_deref())
            .with_finalize(self.config.finalize)
            .build()
    }

    pub(crate) fn system_message(&self) -> Message {
        self.system_message_for(
            self.goal.as_deref(),
            self.structured_goal.as_ref(),
            &self.decisions,
        )
    }

    /// Minimal system message for throwaway model calls (finalize_turn,
    /// summarize, update_memory) — no project_context, no goal, no finalize
    /// instruction. These calls don't need the repo map or session goal; sending
    /// them wastes ~1.5-3K input tokens per call and bloats the uncached portion
    /// of the request.
    pub(crate) fn minimal_system_message(&self) -> Message {
        SystemPrompt::new().build()
    }

    pub(crate) fn refresh_system_message(&mut self) {
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

    /// Set or clear the transient session goal, first clearing any persisted
    /// structured long-horizon goal so it cannot reappear on a later resume.
    pub fn set_transient_goal(&mut self, goal: Option<String>) -> Result<()> {
        self.set_structured_goal(None)?;
        self.set_goal(goal);
        Ok(())
    }

    /// Set or clear a structured long-horizon goal (decomposed into sub-goals).
    /// Only takes effect when `config.long_horizon` is on; when set, the goal's
    /// state is injected into the system prompt each turn so the agent resumes
    /// the active sub-goal. Returns whether it was accepted.
    pub fn set_structured_goal(&mut self, goal: Option<Goal>) -> Result<bool> {
        if !self.config.long_horizon && goal.is_some() {
            return Ok(false);
        }
        if let Some(session) = self.session.as_mut() {
            if let Some(g) = &goal {
                session.record_goal(g)?;
            } else {
                session.clear_goal()?;
            }
        }
        self.structured_goal = goal;
        self.refresh_system_message();
        Ok(true)
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

    pub(crate) fn persist(&mut self) -> Result<()> {
        if let Some(session) = self.session.as_mut() {
            session.record(&self.messages.as_slice()[self.persisted..], self.totals)?;
            self.persisted = self.messages.len();
        }
        Ok(())
    }

    /// Persist the current structured goal (if any) so a `/resume` picks it up
    /// at its active sub-goal. Best-effort: a failure is logged to the UI but
    /// doesn't fail the turn (the goal still lives in-memory for this session).
    pub(crate) fn persist_goal(&mut self, ui: &mut dyn Ui) {
        if let Some(session) = self.session.as_mut()
            && let Some(goal) = &self.structured_goal
            && let Err(err) = session.record_goal(goal)
        {
            ui.status(&format!("(couldn't persist goal: {err})"));
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

/// The tool set an agent advertises for its config: the minimal or full set, plus
/// the `explore`/`delegate` subagent tools when enabled for a top-level agent
/// (never a subagent — depth ≤ 1). Shared by construction and the runtime
/// `/delegate` toggle.
fn advertised_tools(config: &AgentConfig) -> std::sync::Arc<[hi_ai::ToolSpec]> {
    if config.minimal_tools {
        return hi_tools::MINIMAL_TOOL_SPECS.clone().into();
    }
    let mut specs = TOOL_SPECS.clone();
    if !config.is_subagent {
        if config.explore_subagents {
            specs.push(hi_tools::explore_tool_spec());
        }
        if config.write_subagents {
            specs.push(hi_tools::delegate_tool_spec());
        }
    }
    specs.into()
}
