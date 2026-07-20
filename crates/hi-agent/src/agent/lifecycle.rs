//! `Agent` construction, accessors, and persistence: `new`/`resume`/`undo`,
//! the message/usage/context/goal/verify accessors, system-prompt refresh,
//! and `persist`/`persist_goal`/`messages_mut`.

use std::sync::Arc;

use anyhow::{Context, Result};
use hi_ai::{Message, Provider, Role, ToolMode, Usage, provider_error_usage};

use super::tool_selection::advertised_tools;

use crate::compaction::{self, DEFAULT_KEEP_RECENT};
use crate::config::AgentConfig;
use crate::decision::DecisionLog;
use crate::goal::Goal;
use crate::heuristics::humanize_count;
use crate::prompt::SystemPrompt;
use crate::snapshot::SnapshotCache;
use crate::transcript::Transcript;
use crate::ui;
use crate::{
    SessionSink, TurnPhase, TurnTelemetry, Ui, VerificationMode, VerifyStage, WorkspaceRuntime,
};

impl crate::Agent {
    /// Start a fresh session seeded with the system prompt.
    pub fn new(provider: Arc<dyn Provider>, config: AgentConfig) -> Result<Self> {
        Self::with_background_scan(provider, config, None)
    }

    /// Like [`Self::new`] but consumes a pre-started [`BackgroundScan`] so the
    /// initial workspace scan overlaps with all startup work before `Agent::new`
    /// is even called.
    pub fn with_background_scan(
        provider: Arc<dyn Provider>,
        config: AgentConfig,
        scan: Option<crate::change_ledger::BackgroundScan>,
    ) -> Result<Self> {
        let system = SystemPrompt::new()
            .with_workspace_root(&config.workspace_root)
            .with_project_context(config.project_context.as_deref())
            .with_finalize(config.finalize)
            .build();
        Self::with_messages(provider, config, vec![system], 0, scan)
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
    ) -> Result<Self> {
        let persisted = history.len();
        let mut agent = Self::with_messages(provider, config, history, persisted, None)?;
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
        Ok(agent)
    }

    fn with_messages(
        provider: Arc<dyn Provider>,
        config: AgentConfig,
        messages: Vec<Message>,
        persisted: usize,
        scan: Option<crate::change_ledger::BackgroundScan>,
    ) -> Result<Self> {
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
        config.verification.validate()?;
        let runtime = WorkspaceRuntime::new_with_scan(
            &config.workspace_root,
            &config.state_root,
            config.lsp_mode,
            scan,
        )?;
        let tools = advertised_tools(&config, None);
        let last_effective_route = crate::EffectiveModelRoute {
            provider: config.provider_route.clone(),
            model: config.model.clone(),
        };
        // Opt-in: route the skeptic review to a separate OpenAI-compatible
        // endpoint (e.g. a local hi-local server) when configured. Shared with
        // the runtime `/config skeptic-local` toggle so their wiring can't drift.
        let skeptic_provider = crate::local_skeptic::build_skeptic_provider(&config);
        Ok(Self {
            provider,
            skeptic_provider,
            local_skeptic: None,
            config,
            runtime,
            task_context: None,
            last_task_contract: None,
            messages,
            tools,
            session: None,
            delegate_runner: None,
            persisted,
            totals: Usage::default(),
            last_turn_usage: Usage::default(),
            last_user_prompt_tokens: 0,
            last_verify: None,
            context_used: 0,
            checkpoints: Vec::new(),
            last_changed_files: Vec::new(),
            last_file_changes: Vec::new(),
            turn_diff_cache: None,
            turn_stub_scan_cache: None,
            active_turn_ledger_revision: None,
            active_turn_message_start: None,
            auto_skills_written: 0,
            explore_subagents_used: 0,
            delegate_subagents_used: 0,
            last_compat_fallbacks: Vec::new(),
            interrupt: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            last_turn_telemetry: TurnTelemetry::default(),
            last_turn_outcome: None,
            turn_phase: TurnPhase::Setup,
            last_effective_route,
            goal: None,
            structured_goal: None,
            decisions: DecisionLog::default(),
            snapshot_cache: SnapshotCache::default(),
            last_plan: Vec::new(),
            interjections: crate::InterjectionInbox::default(),
            last_rsi_fully_observed: None,
            managed_rsi_context: None,
        })
    }

    /// Installs already-validated managed RSI reference context for the next
    /// one-shot turn. This is deliberately separate from `AgentConfig` so
    /// ordinary agents and read-only subagents cannot inherit it accidentally.
    pub fn set_managed_rsi_context(&mut self, context: Option<String>) {
        self.managed_rsi_context = context;
    }

    /// A cloneable handle for a frontend to push user messages typed while a
    /// turn is running. The running turn drains them at safe points and injects
    /// them as genuine user messages (mid-turn steering).
    pub fn interjection_inbox(&self) -> crate::InterjectionInbox {
        self.interjections.clone()
    }

    /// Revert the file changes the most recent turn made, restoring its git
    /// checkpoint. Returns `None` if there's nothing to undo, else the number of
    /// files restored or removed.
    pub async fn undo(&mut self) -> Result<Option<usize>> {
        let Some(reference) = self.checkpoints.last().cloned() else {
            return Ok(None);
        };
        let (target, expected_current) = hi_tools::checkpoint::parse_reference(&reference)?;
        // If durable stack persistence fails after the restore, put the exact
        // pre-undo tree back before returning. Sealed 0.2 records already carry
        // that immutable post-turn tree; legacy records get a temporary one.
        // This prevents an error from leaving restored files paired with the
        // still-live old checkpoint stack.
        let rollback_checkpoint = if self.session.is_some() {
            if let Some(expected_current) = expected_current {
                Some(expected_current.to_string())
            } else {
                match hi_tools::checkpoint::create_detailed_with_state(
                    self.runtime.root(),
                    self.runtime.state_root(),
                )
                .await
                {
                    hi_tools::checkpoint::CreateResult::Created(id) => Some(id),
                    hi_tools::checkpoint::CreateResult::Unavailable(reason)
                    | hi_tools::checkpoint::CreateResult::Failed(reason) => {
                        anyhow::bail!("cannot prepare transactional undo rollback: {reason}")
                    }
                }
            }
        } else {
            None
        };
        let n = match expected_current {
            Some(expected_current) => {
                hi_tools::checkpoint::restore_sealed_with_state(
                    self.runtime.root(),
                    target,
                    expected_current,
                    self.runtime.state_root(),
                )
                .await?
            }
            // Bare 0.1 checkpoint ids remain readable for migration. New 0.2
            // turns always persist a sealed reference below.
            None => {
                hi_tools::checkpoint::restore_with_state(
                    self.runtime.root(),
                    target,
                    self.runtime.state_root(),
                )
                .await?
            }
        };
        let mut next = self.checkpoints.clone();
        next.pop();
        let persist_result = self
            .session
            .as_mut()
            .map(|session| session.record_checkpoints(&next))
            .unwrap_or(Ok(()));
        if let Err(persist_error) = persist_result {
            let rollback = hi_tools::checkpoint::restore_sealed_with_state(
                self.runtime.root(),
                rollback_checkpoint
                    .as_deref()
                    .context("undo rollback checkpoint was not prepared")?,
                target,
                self.runtime.state_root(),
            )
            .await;
            self.invalidate_snapshot();
            self.runtime.clear_read_cache();
            let reconcile = self.runtime.ledger().reconcile();
            return match (rollback, reconcile) {
                    (Ok(_), Ok(_)) => Err(persist_error.context(
                        "persisting the shortened undo stack failed; workspace rollback succeeded",
                    )),
                    (rollback, reconcile) => Err(persist_error.context(format!(
                        "persisting the shortened undo stack failed; restoring the pre-undo workspace also failed: {}; ledger reconciliation: {}",
                        rollback
                            .err()
                            .map(|error| format!("{error:#}"))
                            .unwrap_or_else(|| "succeeded".to_string()),
                        reconcile
                            .err()
                            .map(|error| format!("{error:#}"))
                            .unwrap_or_else(|| "succeeded".to_string())
                    ))),
                };
        }
        self.checkpoints = next;
        // The working tree just changed under us, so any cached snapshot is now
        // stale. Without this, the next turn reuses pre-undo fingerprints and
        // change detection / verify gating / last_changed_files can be wrong.
        // Clear the read cache too — restore() rewrites files directly, so a read
        // between now and the next turn's clear would otherwise serve pre-undo
        // content.
        self.invalidate_snapshot();
        self.runtime.clear_read_cache();
        // Bring the content ledger back to the restored state and do not report
        // the now-undone effects as the latest workspace changes.
        self.runtime.ledger().reconcile()?;
        self.last_changed_files.clear();
        self.last_file_changes.clear();
        Ok(Some(n))
    }

    /// Attach a session sink that records messages produced from here on.
    pub fn set_session(&mut self, session: Box<dyn SessionSink>) {
        self.session = Some(session);
    }

    /// Detach the current session sink. Used by `--attach --resume-local` to
    /// prevent the pre-existing session sink (pointing to the original local
    /// session file and remote session ID) from recording turns that belong to
    /// the resumed remote session.
    pub fn detach_session(&mut self) {
        self.session = None;
    }

    /// Apply a loaded session state (from remote records or a local JSONL file)
    /// to an existing agent. This is the in-place equivalent of [`Agent::resume`]
    /// — it replaces the transcript, usage, checkpoints, goal, and decisions
    /// without reconstructing the agent.
    pub fn apply_loaded_session(
        &mut self,
        history: Vec<Message>,
        usage: Usage,
        checkpoint_refs: Vec<String>,
        structured_goal: Option<Goal>,
        decisions: DecisionLog,
        plan: Vec<hi_tools::PlanStep>,
    ) {
        let mut messages = crate::Transcript::new(history);
        // Run the same repair pipeline as `with_messages` so a resumed session
        // is cleaned up identically regardless of whether it came from a local
        // JSONL file or remote ipop records.
        messages.strip_finalize_pair();
        messages.strip_trailing_nudges();
        messages.repair_tool_result_ordering();
        messages.repair_invalid_tool_call_arguments();
        messages.repair_provider_invisible_assistant_messages();
        messages.repair_consecutive_assistant_messages();
        messages.repair_consecutive_user_messages();
        // Clamp persisted to the (possibly shorter) transcript length.
        let persisted = messages.len();
        self.messages = messages;
        self.persisted = persisted;
        self.totals = usage;
        self.checkpoints = checkpoint_refs;
        if self.checkpoints.len() > crate::MAX_CHECKPOINTS {
            self.checkpoints
                .drain(0..self.checkpoints.len() - crate::MAX_CHECKPOINTS);
        }
        self.decisions = decisions;
        self.structured_goal = self
            .config
            .long_horizon
            .then_some(structured_goal)
            .flatten();
        // Clear per-turn / transient state from the previous session, matching
        // what `with_messages` initializes to None/empty for a fresh agent.
        self.goal = None;
        self.last_plan = if plan
            .iter()
            .any(|step| step.status != hi_tools::PlanStatus::Done)
        {
            plan
        } else {
            Vec::new()
        };
        self.last_changed_files = Vec::new();
        self.last_turn_telemetry = TurnTelemetry::default();
        self.last_verify = None;
        self.refresh_system_message();
        // The transcript was replaced, so any cached working-tree snapshot is
        // stale. Clear it so the next turn re-snapshots from scratch.
        self.invalidate_snapshot();
        self.runtime.clear_read_cache();
    }

    /// Install an unfinished plan reconstructed by session storage.
    pub fn restore_plan(&mut self, plan: Vec<hi_tools::PlanStep>) {
        self.last_plan = if plan
            .iter()
            .any(|step| step.status != hi_tools::PlanStatus::Done)
        {
            plan
        } else {
            Vec::new()
        };
    }

    pub fn current_plan(&self) -> &[hi_tools::PlanStep] {
        &self.last_plan
    }

    /// Attach the runner that executes write-capable `delegate` subagents. Without
    /// one, the `delegate` tool reports itself unavailable.
    pub fn set_delegate_runner(&mut self, runner: std::sync::Arc<dyn crate::DelegateRunner>) {
        self.delegate_runner = Some(runner);
    }

    /// Set the write-capable `delegate` policy at runtime (`/delegate on|off|risk`)
    /// — re-advertises the tool set accordingly. A [`DelegateRunner`] must be
    /// attached for it to actually run.
    pub fn set_write_subagents(&mut self, policy: crate::WriteSubagentPolicy) {
        self.config.write_subagents = policy;
        self.tools = advertised_tools(&self.config, None);
    }

    /// Convenience for `/delegate on|off` boolean toggles.
    pub fn set_write_subagents_enabled(&mut self, on: bool) {
        self.set_write_subagents(if on {
            crate::WriteSubagentPolicy::On
        } else {
            crate::WriteSubagentPolicy::Off
        });
    }

    pub(crate) fn refresh_tools_for_task(&mut self, task: &str, intent: crate::TaskIntent) {
        self.tools = advertised_tools(&self.config, Some((task, intent)));
    }

    /// Whether `delegate` may be advertised for some tasks (not hard-off).
    pub fn write_subagents_enabled(&self) -> bool {
        self.config.write_subagents.is_enabled()
    }

    /// Current write-subagent policy (`off` / `risk` / `on`).
    pub fn write_subagents_policy(&self) -> crate::WriteSubagentPolicy {
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
                &snapshot.last_plan,
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

    /// Token usage accumulated by the most recent user turn.
    pub fn last_turn_usage(&self) -> &Usage {
        &self.last_turn_usage
    }

    /// Estimated tokens in the raw user prompt for the most recent user turn.
    pub fn last_user_prompt_tokens(&self) -> u64 {
        self.last_user_prompt_tokens
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
        self.runtime.lsp_enabled()
    }

    /// Enable or disable the LSP subsystem at runtime (`/lsp on|off`).
    pub fn set_lsp_enabled(&self, on: bool) {
        self.runtime.set_lsp_enabled(on);
    }

    /// Workspace-local `/lsp status` output.
    pub fn lsp_status_report(&self) -> String {
        let manager = self.runtime.lsp();
        hi_tools::lsp_status_report_for(self.lsp_enabled(), &manager.status_sync())
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

    /// Usage from a *main-conversation* model call: counts toward all totals
    /// and refreshes the context gauge with the request's occupancy.
    pub(crate) fn add_usage(&mut self, usage: Usage) {
        self.add_side_usage(usage);
        let occupancy = if usage.context_occupancy > 0 {
            usage.context_occupancy
        } else {
            usage.effective_input_tokens()
        };
        if occupancy > 0 {
            self.context_used = occupancy;
        }
    }

    /// Usage from a *side* model call (finalize, skeptic, curate, memory,
    /// goal planning, explore children, compaction summarize): counts toward
    /// totals and the turn's spend, but must not touch `context_used` — that
    /// gauge tracks the main conversation's occupancy and drives
    /// auto-compaction, and a ~3K-token side request at the end of a 150K
    /// session would reset it to 2%, silently disabling the next compaction.
    pub(crate) fn add_side_usage(&mut self, usage: Usage) {
        self.totals.add(usage);
        self.last_turn_usage.add(usage);
    }

    pub(crate) fn reset_last_turn_usage(&mut self, user_prompt_tokens: u64) {
        self.last_turn_usage = Usage::default();
        self.last_user_prompt_tokens = user_prompt_tokens;
    }

    pub(crate) fn add_error_usage(&mut self, err: &anyhow::Error) {
        self.add_usage(provider_error_usage(err));
    }

    /// Like [`add_error_usage`] but for a *side* model call (skeptic, curate,
    /// memory, goal planning, finalize, summarize). Books the error's usage
    /// toward totals/turn spend without touching `context_used` — routing a
    /// small side request's input size through `add_usage` would reset the main
    /// conversation's occupancy gauge and silently disable the next
    /// auto-compaction (see [`add_side_usage`]). Providers do attach nonzero
    /// input usage to some errors (e.g. EmptyCompletion/MalformedStream), so
    /// this matters in practice, not just in theory.
    pub(crate) fn add_side_error_usage(&mut self, err: &anyhow::Error) {
        self.add_side_usage(provider_error_usage(err));
    }

    pub(crate) fn emit_usage(&self, ui: &mut dyn Ui) {
        ui.usage(
            self.last_user_prompt_tokens,
            self.last_turn_usage.output_tokens,
            self.context_used,
            self.config.context_window,
            self.last_turn_usage.estimated,
        );
        ui.rate_limits(self.totals.rate_limits);
    }

    /// Number of git checkpoints created so far (for `/undo`).
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Explicit root owned by this agent's workspace runtime.
    pub fn workspace_root(&self) -> &std::path::Path {
        self.runtime.root()
    }

    /// Snapshot this agent runtime's background handles for cancellable turns.
    pub fn background_process_ids(&self) -> Vec<String> {
        self.runtime.background().ids()
    }

    /// Kill only background processes this agent started after `before`.
    pub fn kill_background_processes_started_after(&self, before: &[String]) -> usize {
        self.runtime.background().kill_started_after(before)
    }

    /// Stop every background process owned by this agent runtime, plus any
    /// auto-managed local skeptic server, on session shutdown.
    pub fn kill_background_processes(&self) {
        self.runtime.background().kill_all();
        self.stop_local_skeptic_server();
    }

    /// Finalize a turn whose future was cancelled by its frontend. Reconcile
    /// after rollback/cleanup so reports contain the exact surviving effects
    /// instead of a fabricated empty list.
    pub fn finalize_cancelled_turn(&mut self) -> Result<crate::TurnOutcome> {
        if let Some(start) = self.active_turn_message_start.take() {
            self.truncate_messages(start);
        }
        self.runtime.ledger().reconcile()?;
        let baseline = self
            .active_turn_ledger_revision
            .take()
            .unwrap_or_else(|| self.runtime.ledger().revision());
        let changes = self.runtime.ledger().changes_since(baseline);
        self.last_changed_files = changes.iter().map(|change| change.path.clone()).collect();
        self.last_file_changes = changes;
        self.last_verify = None;
        let outcome = crate::TurnOutcome {
            status: crate::TurnStatus::Cancelled,
            verification: crate::VerificationStatus::Unverified,
            review: crate::ReviewStatus::NotRequired,
            stop_reason: crate::TurnStopReason::Cancelled,
            changed_files: self.last_changed_files.clone(),
            verified_workspace_revision: None,
            effective_route: self.last_effective_route.clone(),
        };
        self.last_turn_outcome = Some(outcome.clone());
        let _ = self.persist();
        Ok(outcome)
    }

    /// Reconcile and type a turn that escaped through an infrastructure or
    /// provider error before the normal common finalizer ran. Frontends call
    /// this before writing reports so late UI/session effects are never
    /// replaced by a fabricated empty change list.
    pub fn finalize_failed_turn(&mut self) -> crate::TurnOutcome {
        let baseline = self
            .active_turn_ledger_revision
            .take()
            .unwrap_or_else(|| self.runtime.ledger().revision());
        let _ = self.runtime.ledger().reconcile();
        let changes = self.runtime.ledger().changes_since(baseline);
        self.last_changed_files = changes.iter().map(|change| change.path.clone()).collect();
        self.last_file_changes = changes;
        self.last_verify = None;
        self.active_turn_message_start = None;
        let route = self.last_effective_route.clone();
        let outcome = crate::TurnOutcome::infrastructure_failure(
            route.model,
            route.provider,
            self.last_changed_files.clone(),
        );
        self.last_turn_outcome = Some(outcome.clone());
        outcome
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
        let combined_context = match (
            self.config.project_context.as_deref(),
            self.task_context.as_deref(),
        ) {
            (Some(project), Some(task)) => Some(format!("{project}\n\n{task}")),
            (Some(project), None) => Some(project.to_string()),
            (None, Some(task)) => Some(task.to_string()),
            (None, None) => None,
        };
        SystemPrompt::new()
            .with_workspace_root(self.runtime.root())
            .with_project_context(combined_context.as_deref())
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
    /// instruction. These calls don't need the task index or session goal; sending
    /// them wastes ~1.5-3K input tokens per call and bloats the uncached portion
    /// of the request.
    pub(crate) fn minimal_system_message(&self) -> Message {
        SystemPrompt::new()
            .with_workspace_root(self.runtime.root())
            .build()
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

    /// Pause or resume the structured goal without losing progress: a paused goal
    /// is dropped from the system prompt and the driver leaves it alone, but its
    /// sub-goal progress is retained and persisted so `/goal resume` picks up
    /// exactly where it left off. Returns whether there was a goal to update.
    pub fn set_goal_paused(&mut self, paused: bool) -> bool {
        let snapshot = match self.structured_goal.as_mut() {
            Some(goal) => {
                goal.paused = paused;
                goal.clone()
            }
            None => return false,
        };
        if let Some(session) = self.session.as_mut() {
            let _ = session.record_goal(&snapshot);
        }
        self.refresh_system_message();
        true
    }

    /// Turn the `/goal team` skeptic gate on/off for the active goal. Persists with
    /// the goal (so a resumed goal remembers it) and refreshes the system message.
    /// Returns `false` if there's no active goal.
    pub fn set_goal_team(&mut self, on: bool) -> bool {
        let snapshot = match self.structured_goal.as_mut() {
            Some(goal) => {
                goal.team = on;
                goal.clone()
            }
            None => return false,
        };
        if let Some(session) = self.session.as_mut() {
            let _ = session.record_goal(&snapshot);
        }
        self.refresh_system_message();
        true
    }

    /// Set (or clear, with `None`) a ceiling on how many sub-goals the goal's plan
    /// may grow to. `None` is the default — no limit, the plan keeps expanding as the
    /// agent discovers work. Persisted with the goal. Returns whether there was a
    /// goal to update.
    pub fn set_goal_step_limit(&mut self, limit: Option<usize>) -> bool {
        let snapshot = match self.structured_goal.as_mut() {
            Some(goal) => {
                goal.step_limit = limit;
                goal.clone()
            }
            None => return false,
        };
        if let Some(session) = self.session.as_mut() {
            let _ = session.record_goal(&snapshot);
        }
        true
    }

    /// One-line goal summary for status surfaces: the structured goal's
    /// progress ("objective — 2/7 sub-goals done", with a paused marker) when one
    /// is set, else the transient goal string, else "off".
    pub fn goal_summary(&self) -> String {
        if let Some(g) = &self.structured_goal {
            let done = g
                .sub_goals
                .iter()
                .filter(|s| s.status == crate::GoalStatus::Done)
                .count();
            let paused = if g.paused { " · paused" } else { "" };
            let skeptic = if g.team {
                format!(
                    " · skeptic: {} unavailable, last {}",
                    g.skeptic_unavailable,
                    g.last_skeptic_status
                        .map(|status| format!("{status:?}"))
                        .unwrap_or_else(|| "not run".into())
                )
            } else {
                String::new()
            };
            return format!(
                "{} — {}/{} sub-goals done{paused}{skeptic}",
                g.objective,
                done,
                g.sub_goals.len()
            );
        }
        self.goal.clone().unwrap_or_else(|| "off".to_string())
    }

    /// Whether long-horizon agency is on (the `long_horizon` config flag), so
    /// frontends can branch `/goal` between the structured goal and the
    /// transient goal string.
    pub fn long_horizon(&self) -> bool {
        self.config.long_horizon
    }

    /// Whether a planner model is configured for `/goal` decomposition
    /// ([`decompose_goal`](Self::decompose_goal)).
    pub fn has_planner(&self) -> bool {
        self.config.planner_model.is_some()
    }

    /// The model the `/goal team` review gate uses: `skeptic_model` when
    /// configured, otherwise the session model. Never empty — the gate works
    /// with zero configuration.
    pub fn effective_skeptic_model(&self) -> &str {
        self.config
            .skeptic_model
            .as_deref()
            .unwrap_or(&self.config.model)
    }

    /// Whether the most recent turn's verification passed (None if not run).
    pub fn last_verify(&self) -> Option<bool> {
        self.last_verify
    }

    /// Files whose content or presence changed in the most recent turn.
    pub fn last_changed_files(&self) -> &[String] {
        &self.last_changed_files
    }

    /// Exact structured file changes reported by tools during the last turn.
    pub fn last_file_changes(&self) -> &[hi_tools::FileChange] {
        &self.last_file_changes
    }

    /// Merge repeated edits to one path into a turn-level before/after record.
    pub(crate) fn record_tool_effects(&mut self, effects: &hi_tools::ToolEffects) -> Result<()> {
        self.runtime.ledger().record_tool_effects(effects)?;
        if effects.mutation_applied {
            if let Some(contract) = self.last_task_contract.as_mut() {
                contract.observe_mutation();
            }
            self.runtime.clear_repo_map_cache();
            self.runtime.invalidate_context();
        }
        self.merge_file_changes(&effects.file_changes);
        Ok(())
    }

    pub(crate) fn reconcile_workspace_changes(&mut self) -> Result<()> {
        let changes = self.runtime.ledger().reconcile()?;
        if !changes.is_empty() {
            if let Some(contract) = self.last_task_contract.as_mut() {
                contract.observe_mutation();
            }
            self.runtime.clear_repo_map_cache();
            self.runtime.invalidate_context();
            self.merge_file_changes(&changes);
        }
        Ok(())
    }

    fn merge_file_changes(&mut self, changes: &[hi_tools::FileChange]) {
        for change in changes {
            if let Some(index) = self
                .last_file_changes
                .iter()
                .position(|existing| existing.path == change.path)
            {
                let existing = &self.last_file_changes[index];
                if existing.before_digest == change.after_digest
                    && existing.before_mode == change.after_mode
                {
                    self.last_file_changes.remove(index);
                    continue;
                }
                let existing = &mut self.last_file_changes[index];
                existing.after_digest = change.after_digest.clone();
                existing.after_len = change.after_len;
                existing.after_mode = change.after_mode;
                existing.kind = match (
                    existing.before_digest.is_some(),
                    change.after_digest.is_some(),
                ) {
                    (false, true) => hi_tools::FileChangeKind::Create,
                    (true, false) => hi_tools::FileChangeKind::Delete,
                    (true, true) => hi_tools::FileChangeKind::Modify,
                    (false, false) => change.kind,
                };
            } else {
                self.last_file_changes.push(change.clone());
            }
        }
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

    /// Actual deterministic verification executions retained for the latest
    /// turn, including failed turns that ended during later reconciliation or
    /// provider recovery.
    pub fn last_verification_executions(&self) -> &[crate::VerificationExecution] {
        &self.last_turn_telemetry.verification_executions
    }

    /// Typed outcome of the most recent successfully finalized turn.
    pub fn last_turn_outcome(&self) -> Option<&crate::TurnOutcome> {
        self.last_turn_outcome.as_ref()
    }

    pub fn last_effective_route(&self) -> &crate::EffectiveModelRoute {
        &self.last_effective_route
    }

    /// Provider label supplied by the frontend for the effective route.
    pub fn provider_route(&self) -> Option<&str> {
        self.config.provider_route.as_deref()
    }

    /// The tool mode currently configured for this session.
    pub fn tool_mode(&self) -> ToolMode {
        self.config.tool_mode
    }

    /// A read-only snapshot of all live agent settings for `/config show`.
    pub fn config_snapshot(&self) -> crate::ConfigSnapshot {
        let c = &self.config;
        crate::ConfigSnapshot {
            model: c.model.clone(),
            provider_route: c.provider_route.clone().unwrap_or_default(),
            max_tokens: if c.max_tokens_explicit {
                format!("{} (explicit)", c.max_tokens)
            } else {
                c.max_tokens.to_string()
            },
            thinking_budget: c
                .thinking_budget
                .map(|n| n.to_string())
                .unwrap_or_else(|| "off".into()),
            reasoning_effort: c
                .reasoning_effort
                .map(|e| e.as_str().to_string())
                .unwrap_or_else(|| "off".into()),
            temperature: c
                .temperature
                .map(|t| t.to_string())
                .unwrap_or_else(|| "default".into()),
            max_steps: self.max_steps_setting(),
            tool_mode: c.tool_mode.label().to_string(),
            compat: c.compat.label().to_string(),
            verify: self.verify_summary(),
            review: c.review.label().to_string(),
            lsp: c.lsp_mode.label().to_string(),
            tool_set: c.tool_set.label().to_string(),
            auto_compact: if c.auto_compact {
                format!("on (≥{}%)", c.auto_compact_percent)
            } else {
                "off".into()
            },
            proactive_verify: c.proactive_verify,
            read_only_preflight: c.read_only_preflight,
            long_horizon: c.long_horizon,
            confirm_edits: c.confirm_edits,
            curate_skills: c.curate_skills,
            explore_subagents: c.explore_subagents,
            write_subagents: c.write_subagents.as_str().into(),
            planner_model: c.planner_model.clone().unwrap_or_else(|| "off".into()),
            skeptic_model: c.skeptic_model.clone().unwrap_or_else(|| "off".into()),
            moe_streaming: match std::env::var("HI_MLX_EXPERT_STREAMING").as_deref() {
                Ok("0") => "off".into(),
                Ok(_) => "on".into(),
                Err(_) => "auto".into(),
            },
        }
    }

    /// Whether any verification stage is configured.
    pub fn verify_is_on(&self) -> bool {
        !matches!(self.config.verification, VerificationMode::Disabled)
    }

    /// A one-line summary of the verification pipeline (`"off"` when none) —
    /// e.g. `"cargo check → cargo test"`.
    pub fn verify_summary(&self) -> String {
        match &self.config.verification {
            VerificationMode::Disabled => "off".to_string(),
            VerificationMode::Auto => {
                let stages = self
                    .config
                    .verification
                    .resolved_stages(self.runtime.root());
                if stages.is_empty() {
                    "auto (no pipeline detected)".to_string()
                } else {
                    format!(
                        "auto: {}",
                        stages
                            .iter()
                            .map(|s| s.command.as_str())
                            .collect::<Vec<_>>()
                            .join(" → ")
                    )
                }
            }
            VerificationMode::Explicit(stages) => stages
                .iter()
                .map(|s| s.command.as_str())
                .collect::<Vec<_>>()
                .join(" → "),
        }
    }

    /// Verification mode configured for subsequent turns.
    pub fn verification_mode(&self) -> &VerificationMode {
        &self.config.verification
    }

    /// Stages resolved for the current workspace (empty when disabled or when
    /// automatic detection found no applicable pipeline).
    pub fn resolved_verification_stages(&self) -> Vec<VerifyStage> {
        self.config
            .verification
            .resolved_stages(self.runtime.root())
    }

    /// The models the current provider/endpoint actually serves (via its
    /// `/models` route), with any live metadata — for the `/model` picker and
    /// the live context/price/health wiring. Empty if unsupported.
    pub async fn list_models(&self) -> Result<Vec<hi_ai::ServedModel>> {
        self.provider.list_models().await
    }

    /// Set or clear a single custom verify command (from `/verify <cmd>`),
    /// replacing any configured pipeline with one stage (or clearing it).
    pub fn set_verify_command(&mut self, cmd: Option<String>) -> Result<()> {
        let verification = match cmd {
            Some(c) => VerificationMode::Explicit(vec![VerifyStage::new("verify", c)]),
            None => VerificationMode::Disabled,
        };
        verification.validate()?;
        self.config.verification = verification;
        Ok(())
    }

    /// Replace the verification pipeline (from auto-detection).
    pub fn set_verify_pipeline(&mut self, stages: Vec<VerifyStage>) -> Result<()> {
        let verification = VerificationMode::Explicit(stages);
        verification.validate()?;
        self.config.verification = verification;
        Ok(())
    }

    /// The reasoning effort applied to main-turn requests (`None` = off, i.e. no
    /// `reasoning_effort` sent and the endpoint's own default is used).
    pub fn reasoning_effort(&self) -> Option<hi_ai::ReasoningEffort> {
        self.config.reasoning_effort
    }

    /// Set (or clear, with `None`) the reasoning effort for subsequent turns.
    /// Applies to main-turn requests on OpenAI-compatible endpoints that accept
    /// `reasoning_effort`; the Anthropic adapter and non-supporting endpoints
    /// ignore it. Safe to call between turns (like the other `/`-command setters).
    pub fn set_reasoning_effort(&mut self, effort: Option<hi_ai::ReasoningEffort>) {
        self.config.reasoning_effort = effort;
    }

    /// The sampling temperature applied to requests (`None` = provider default).
    pub fn temperature(&self) -> Option<f32> {
        self.config.temperature
    }

    /// Set (or clear, with `None`) the sampling temperature for subsequent turns.
    pub fn set_temperature(&mut self, temperature: Option<f32>) {
        self.config.temperature = temperature;
    }

    /// Human-readable live step-limit setting. `auto` uses the intent-aware
    /// defaults; `off` uses no practical per-turn cap.
    pub fn max_steps_setting(&self) -> String {
        if !self.config.max_steps_explicit {
            "auto".to_string()
        } else if self.config.max_steps == u32::MAX {
            "off".to_string()
        } else {
            self.config.max_steps.to_string()
        }
    }

    pub fn max_tool_calls_limit(&self) -> u32 {
        self.config.max_tool_calls
    }

    /// Set a fixed per-turn step cap, or disable the cap with `None`.
    pub fn set_max_steps_limit(&mut self, limit: Option<u32>) {
        self.config.max_steps = limit.unwrap_or(u32::MAX).max(1);
        self.config.max_steps_explicit = true;
    }

    /// Restore intent-aware automatic step limits for subsequent turns.
    pub fn set_max_steps_auto(&mut self) {
        self.config.max_steps_explicit = false;
    }

    pub fn rsi_status(&self) -> (&'static str, &'static str, Option<bool>) {
        let requested = if self.config.rsi_enabled { "on" } else { "off" };
        let mode = if self.config.rsi_managed {
            "managed"
        } else if self.config.rsi_enabled {
            "remote"
        } else {
            "off"
        };
        (requested, mode, self.last_rsi_fully_observed)
    }

    pub fn rsi_maximum_cost_microusd(&self) -> Option<u64> {
        self.config
            .rsi_control
            .as_ref()
            .map(|control| control.maximum_cost_microusd())
    }

    pub fn rsi_channel(&self) -> &'static str {
        self.config
            .rsi_control
            .as_ref()
            .map_or("stable", |control| control.channel())
    }

    pub fn set_rsi_channel(&mut self, channel: crate::command::RsiChannel) -> Result<()> {
        let control = self
            .config
            .rsi_control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.set_channel(channel.as_str())
    }

    pub async fn rsi_public_status(&self) -> Result<String> {
        let control = self
            .config
            .rsi_control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.status().await
    }

    pub fn set_rsi_maximum_cost_microusd(&mut self, value: u64) -> Result<()> {
        anyhow::ensure!(
            (1..=15_000_000).contains(&value),
            "RSI spend limit must be greater than $0 and no more than $15"
        );
        let control = self
            .config
            .rsi_control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.set_maximum_cost_microusd(value)
    }

    pub fn set_rsi_enabled(&mut self, enabled: bool) -> Result<()> {
        anyhow::ensure!(
            !self.config.rsi_managed || enabled,
            "managed RSI cannot be disabled"
        );
        if enabled && !self.config.rsi_managed {
            anyhow::ensure!(
                self.config.rsi_remote_switch.is_some(),
                "remote RSI requires PIPENETWORK_API_KEY or an active Pipe provider key"
            );
        }
        self.config.rsi_enabled = enabled;
        if let Some(switch) = &self.config.rsi_remote_switch {
            switch.store(enabled, std::sync::atomic::Ordering::SeqCst);
        }
        Ok(())
    }

    pub async fn set_rsi_enabled_validated(&mut self, enabled: bool) -> Result<()> {
        let control = self.config.rsi_control.clone();
        if enabled && !self.config.rsi_managed {
            let control = control
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
            control.validate().await?;
        }
        if !self.config.rsi_managed
            && let Some(control) = &control
        {
            control.persist_enabled(enabled)?;
        }
        self.set_rsi_enabled(enabled)
    }

    pub async fn rsi_command(&self, argument: &str) -> Result<String> {
        let control = self
            .config
            .rsi_control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.command(argument).await
    }

    pub fn set_last_rsi_fully_observed(&mut self, observed: Option<bool>) {
        self.last_rsi_fully_observed = observed;
    }

    pub(crate) fn persist(&mut self) -> Result<()> {
        if let Some(session) = self.session.as_mut() {
            // Clamp the cursor: transcript-shrinking ops (`strip_trailing_nudges`,
            // `strip_finalize_pair`) pop messages without adjusting `persisted`,
            // so after a mid-turn persist that already recorded up to a
            // now-popped message, `persisted` can exceed the current length.
            // Slicing `[persisted..]` would then panic; clamp so we simply record
            // nothing new instead of crashing the session.
            let start = self.persisted.min(self.messages.len());
            session.record(&self.messages.as_slice()[start..], self.totals)?;
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
