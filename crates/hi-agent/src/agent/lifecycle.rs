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
use crate::{SessionSink, TurnTelemetry, Ui, VerificationMode, VerifyStage, WorkspaceRuntime};

impl crate::Agent {
    /// Start a fresh session seeded with the system prompt.
    pub fn new(provider: Arc<dyn Provider>, config: AgentConfig) -> Result<Self> {
        Self::with_background_scan(provider, config, None)
    }

    /// Install an in-process lifecycle extension registry. Contributors are
    /// fired at turn start/done/error/abort. Call after `new`/`resume` and
    /// before the first `run_turn`.
    pub fn with_extension_registry(mut self, registry: hi_agent_lifecycle::ExtensionRegistry) -> Self {
        self.extensions = Some(registry);
        self
    }

    /// The installed in-process extension registry, if any.
    pub fn extensions(&self) -> Option<&hi_agent_lifecycle::ExtensionRegistry> {
        self.extensions.as_ref()
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
            .with_workspace_root(&config.paths.workspace_root)
            .with_project_context(config.memory.project_context.as_deref())
            .with_finalize(config.memory.finalize)
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
        agent.workspace.checkpoints = checkpoint_refs;
        if agent.workspace.checkpoints.len() > crate::MAX_CHECKPOINTS {
            agent
                .workspace
                .checkpoints
                .drain(0..agent.workspace.checkpoints.len() - crate::MAX_CHECKPOINTS);
        }
        agent.decisions = decisions;
        agent.goals.structured = agent
            .config
            .subagents
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
        config.gates.verification.validate()?;
        let runtime = WorkspaceRuntime::new_with_scan(
            &config.paths.workspace_root,
            &config.paths.state_root,
            config.gates.lsp_mode,
            scan,
        )?;
        let tools = advertised_tools(&config, None);
        let last_effective_route = crate::EffectiveModelRoute {
            provider: config.routing.provider_route.clone(),
            model: config.routing.model.clone(),
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
            task: crate::domain::TaskContextState::default(),
            messages,
            tools,
            session: None,
            persisted,
            totals: Usage::default(),
            report: crate::domain::TurnReportState::new(last_effective_route),
            workspace: crate::domain::WorkspaceTurnState::default(),
            subagents: crate::domain::SubagentSessionState::default(),
            bg_tasks: hi_tools::BackgroundTaskRegistry::new(),
            interrupt: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            goals: crate::domain::GoalState::default(),
            decisions: DecisionLog::default(),
            snapshot_cache: SnapshotCache::default(),
            interjections: crate::InterjectionInbox::default(),
            btw_answer_pending: false,
            pending_block: None,
            rsi_observe: crate::domain::RsiObserveState::default(),
            plan_mode: false,
            permission_mode: crate::PermissionMode::default(),
            turn_count: 0,
            extensions: None,
        })
    }

    /// Installs already-validated managed RSI reference context for the next
    /// one-shot turn. This is deliberately separate from `AgentConfig` so
    /// ordinary agents and read-only subagents cannot inherit it accidentally.
    pub fn set_managed_rsi_context(&mut self, context: Option<String>) {
        self.rsi_observe.set_managed_context(context);
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
        let Some(reference) = self.workspace.checkpoints.last().cloned() else {
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
        let mut next = self.workspace.checkpoints.clone();
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
            let reconcile = self.runtime.reconcile_ledger_async().await;
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
        self.workspace.checkpoints = next;
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
        self.runtime.reconcile_ledger_async().await?;
        self.workspace.last_changed_files.clear();
        self.workspace.last_file_changes.clear();
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
        self.workspace.checkpoints = checkpoint_refs;
        if self.workspace.checkpoints.len() > crate::MAX_CHECKPOINTS {
            self.workspace
                .checkpoints
                .drain(0..self.workspace.checkpoints.len() - crate::MAX_CHECKPOINTS);
        }
        self.decisions = decisions;
        self.goals.structured = self
            .config
            .subagents
            .long_horizon
            .then_some(structured_goal)
            .flatten();
        // Clear per-turn / transient state from the previous session, matching
        // what `with_messages` initializes to None/empty for a fresh agent.
        self.goals.free_text = None;
        self.goals.set_plan_if_pending(plan);
        self.workspace.last_changed_files = Vec::new();
        self.report.last_turn_telemetry = TurnTelemetry::default();
        self.report.last_verify = None;
        self.refresh_system_message();
        // The transcript was replaced, so any cached working-tree snapshot is
        // stale. Clear it so the next turn re-snapshots from scratch.
        self.invalidate_snapshot();
        self.runtime.clear_read_cache();
    }

    /// Install an unfinished plan reconstructed by session storage.
    pub fn restore_plan(&mut self, plan: Vec<hi_tools::PlanStep>) {
        self.goals.set_plan_if_pending(plan);
    }

    pub fn current_plan(&self) -> &[hi_tools::PlanStep] {
        self.goals.plan()
    }

    /// Whether `/plan` mode is active (frontends should prefer read-only tools).
    pub fn plan_mode(&self) -> bool {
        self.plan_mode
    }

    pub fn set_plan_mode(&mut self, on: bool) {
        self.plan_mode = on;
        if on {
            // Plan mode pairs with ask-style caution for accidental mutations.
            if self.permission_mode == crate::PermissionMode::Always {
                self.set_permission_mode(crate::PermissionMode::Ask);
            }
            // Advertise tools as if the next task were read-only (no mutations).
            self.tools = advertised_tools(&self.config, Some(("", crate::TaskIntent::ReadOnly)));
        } else {
            self.tools = advertised_tools(&self.config, None);
        }
    }

    pub fn permission_mode(&self) -> crate::PermissionMode {
        self.permission_mode
    }

    /// Apply the permission ladder to live gates (`confirm_edits` / checkpoint).
    pub fn set_permission_mode(&mut self, mode: crate::PermissionMode) {
        self.permission_mode = mode;
        match mode {
            crate::PermissionMode::Ask => {
                self.config.gates.confirm_edits = true;
                self.config.gates.allow_no_checkpoint = false;
            }
            crate::PermissionMode::Auto => {
                // Auto keeps the confirmation pipeline enabled; frontends may
                // auto-approve only `ConfirmationRequest::safe_for_auto()` and
                // surface everything else. Checkpoints remain mandatory.
                self.config.gates.confirm_edits = true;
                self.config.gates.allow_no_checkpoint = false;
            }
            crate::PermissionMode::Always => {
                self.config.gates.confirm_edits = false;
                self.config.gates.allow_no_checkpoint = true;
            }
        }
    }

    /// Rewind conversation to just before user turn `n` (1-based). Does not
    /// restore files — pair with `/undo` for workspace rollback.
    pub fn rewind_to_user_turn(&mut self, turn_n: usize) -> Result<String> {
        let len = crate::rewind_len_before_user_turn(self.messages(), turn_n)?;
        let before = self.messages().len();
        self.truncate_messages_durable(len)?;
        let after = self.messages().len();
        Ok(format!(
            "rewound to before user turn {turn_n} (messages {before} → {after}). workspace files unchanged — /undo reverts the last turn's edits if needed."
        ))
    }

    /// Attach the runner that executes write-capable `delegate` subagents. Without
    /// one, the `delegate` tool reports itself unavailable.
    pub fn set_delegate_runner(&mut self, runner: std::sync::Arc<dyn crate::DelegateRunner>) {
        self.subagents.delegate_runner = Some(runner);
    }

    /// Set the write-capable `delegate` policy at runtime (`/delegate on|off|risk`)
    /// — re-advertises the tool set accordingly. A [`DelegateRunner`] must be
    /// attached for it to actually run.
    pub fn set_write_subagents(&mut self, policy: crate::WriteSubagentPolicy) {
        self.config.subagents.write_subagents = policy;
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
        let intent = if self.plan_mode {
            crate::TaskIntent::ReadOnly
        } else {
            intent
        };
        self.tools = advertised_tools(&self.config, Some((task, intent)));
    }

    /// Whether `delegate` may be advertised for some tasks (not hard-off).
    pub fn write_subagents_enabled(&self) -> bool {
        self.config.subagents.write_subagents.is_enabled()
    }

    /// Current write-subagent policy (`off` / `risk` / `on`).
    pub fn write_subagents_policy(&self) -> crate::WriteSubagentPolicy {
        self.config.subagents.write_subagents
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
        let (goal, structured_goal, last_plan) = self.goals.snapshot_triple();
        crate::AgentStateSnapshot {
            goal,
            structured_goal,
            decisions: self.decisions.clone(),
            last_plan,
        }
    }

    /// Live-only restore of a previously captured state snapshot. Used as a
    /// fallback after a failed durable discard so the current process still
    /// reflects the user's explicit interrupt.
    pub fn restore_state_snapshot(&mut self, snapshot: &crate::AgentStateSnapshot) {
        self.goals.restore_triple(
            snapshot.goal.clone(),
            snapshot.structured_goal.clone(),
            snapshot.last_plan.clone(),
        );
        self.decisions = snapshot.decisions.clone();
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
            .subagents
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
        self.goals.restore_triple(
            snapshot.goal.clone(),
            structured_goal,
            snapshot.last_plan.clone(),
        );
        self.decisions = snapshot.decisions.clone();
        Ok(())
    }

    /// Cumulative token usage across the session.
    pub fn totals(&self) -> &Usage {
        &self.totals
    }

    /// Token usage accumulated by the most recent user turn.
    pub fn last_turn_usage(&self) -> &Usage {
        &self.report.last_turn_usage
    }

    /// Estimated tokens in the raw user prompt for the most recent user turn.
    pub fn last_user_prompt_tokens(&self) -> u64 {
        self.report.last_user_prompt_tokens
    }

    /// The context-window occupancy, as last reported by the provider.
    pub fn context_used(&self) -> u64 {
        self.report.context_used
    }

    /// The configured context window, if known.
    pub fn context_window(&self) -> Option<u32> {
        self.config.routing.context_window
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
        let window = self.config.routing.context_window;
        let total_est = compaction::estimate_tokens(messages);
        let mut out = String::new();
        if let Some(w) = window
            && w > 0
        {
            let pct = (self.report.context_used * 100 / u64::from(w)).min(100);
            out.push_str(&format!(
                "context: {} / {} tokens ({}% used)\n",
                humanize_count(self.report.context_used),
                humanize_count(u64::from(w)),
                pct,
            ));
            out.push_str(&format!(
                "  estimated history: {} tokens\n",
                humanize_count(total_est),
            ));
            // How many turns until compaction triggers?
            let threshold = u64::from(w) * self.config.memory.auto_compact_percent / 100;
            if self.report.context_used < threshold {
                let headroom = threshold.saturating_sub(self.report.context_used);
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
                humanize_count(self.report.context_used),
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
            self.config.memory.compaction
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
            self.report.context_used = occupancy;
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
        self.report.last_turn_usage.add(usage);
    }

    pub(crate) fn reset_last_turn_usage(&mut self, user_prompt_tokens: u64) {
        self.report.last_turn_usage = Usage::default();
        self.report.last_user_prompt_tokens = user_prompt_tokens;
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
            self.report.last_user_prompt_tokens,
            self.report.last_turn_usage.output_tokens,
            self.report.context_used,
            self.config.routing.context_window,
            self.report.last_turn_usage.estimated,
        );
        ui.rate_limits(self.totals.rate_limits);
    }

    /// Number of git checkpoints created so far (for `/undo`).
    pub fn checkpoint_count(&self) -> usize {
        self.workspace.checkpoints.len()
    }

    /// Explicit root owned by this agent's workspace runtime.
    pub fn workspace_root(&self) -> &std::path::Path {
        self.runtime.root()
    }

    /// Snapshot this agent runtime's background handles for cancellable turns.
    pub fn background_process_ids(&self) -> Vec<String> {
        self.runtime.background().ids()
    }

    /// A read-only snapshot of this session's background jobs `(id, command,
    /// status)` — used by the `/btw` session snapshot so the model can answer
    /// "what jobs are running / did my task finish" without polling.
    pub(crate) fn background_snapshot(&self) -> Vec<(String, String, String)> {
        self.runtime.background().snapshot()
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
        // Background subagent tasks are cleaned up via BackgroundTaskRegistry's
        // Drop impl when the agent is dropped. The async `kill_all` method can
        // be called from async cleanup paths if needed.
    }

    /// Single entry point for abnormal turn teardown (cancel / infrastructure fail).
    ///
    /// Owns turn-scoped background kill via [`WorkspaceTurnState::active_turn_background_baseline`]
    /// (taken once — second call is a no-op). Frontends should prefer this over
    /// ad-hoc kill + finalize sequences.
    ///
    /// Normal successful turns clear baselines inside `run_turn` and must not call this.
    pub async fn cleanup_turn(
        &mut self,
        kind: crate::TurnCleanupKind,
    ) -> Result<crate::TurnCleanupResult> {
        match kind {
            crate::TurnCleanupKind::Cancel { session } => {
                let killed = self.take_and_kill_turn_backgrounds();
                match session {
                    crate::SessionRollback::AlreadyApplied => {
                        // Frontend already rewound transcript/goals; don't truncate again.
                        let _ = self.workspace.active_turn_message_start.take();
                    }
                    crate::SessionRollback::AgentOwned {
                        checkpoint_count_before,
                    } => {
                        if self.checkpoint_count() > checkpoint_count_before
                            && let Err(err) = self.undo().await
                        {
                            eprintln!(
                                "hi-agent: couldn't roll back cancelled workspace edits: {err:#}"
                            );
                        }
                        if let Some(start) = self.workspace.active_turn_message_start.take() {
                            self.truncate_messages(start);
                        }
                    }
                }
                let outcome = self.finalize_cancelled_turn_inner()?;
                Ok(crate::TurnCleanupResult {
                    outcome,
                    killed_backgrounds: killed,
                })
            }
            crate::TurnCleanupKind::Fail => {
                let killed = self.take_and_kill_turn_backgrounds();
                let outcome = self.finalize_failed_turn_inner();
                Ok(crate::TurnCleanupResult {
                    outcome,
                    killed_backgrounds: killed,
                })
            }
        }
    }

    /// Finalize a cancelled turn. Prefer [`Self::cleanup_turn`] so background kill
    /// and session rollback stay consistent across frontends.
    pub fn finalize_cancelled_turn(&mut self) -> Result<crate::TurnOutcome> {
        let _ = self.take_and_kill_turn_backgrounds();
        self.finalize_cancelled_turn_inner()
    }

    /// Finalize a failed turn. Prefer [`Self::cleanup_turn`]([`TurnCleanupKind::Fail`]).
    pub fn finalize_failed_turn(&mut self) -> crate::TurnOutcome {
        let _ = self.take_and_kill_turn_backgrounds();
        self.finalize_failed_turn_inner()
    }

    fn finalize_cancelled_turn_inner(&mut self) -> Result<crate::TurnOutcome> {
        // Message truncate only if still set (AlreadyApplied path takes it first).
        if let Some(start) = self.workspace.active_turn_message_start.take() {
            self.truncate_messages(start);
        }
        self.runtime.ledger().reconcile()?;
        let baseline = self
            .workspace
            .active_turn_ledger_revision
            .take()
            .unwrap_or_else(|| self.runtime.ledger().revision());
        let changes = self.runtime.ledger().changes_since(baseline);
        self.workspace.record_changes(changes, true);
        self.report.clear_verify();
        self.workspace.clear_active_baselines();
        let outcome = crate::TurnOutcome {
            status: crate::TurnStatus::Cancelled,
            verification: crate::VerificationStatus::Unverified,
            review: crate::ReviewStatus::NotRequired,
            stop_reason: crate::TurnStopReason::Cancelled,
            changed_files: self.workspace.last_changed_files.clone(),
            verified_workspace_revision: None,
            effective_route: self.report.last_effective_route.clone(),
        };
        self.report.set_outcome(outcome.clone());
        let _ = self.persist();
        Ok(outcome)
    }

    fn finalize_failed_turn_inner(&mut self) -> crate::TurnOutcome {
        let baseline = self
            .workspace
            .active_turn_ledger_revision
            .take()
            .unwrap_or_else(|| self.runtime.ledger().revision());
        let _ = self.runtime.ledger().reconcile();
        let changes = self.runtime.ledger().changes_since(baseline);
        self.workspace.record_changes(changes, true);
        self.report.clear_verify();
        self.workspace.clear_active_baselines();
        let route = self.report.last_effective_route.clone();
        let outcome = crate::TurnOutcome::infrastructure_failure(
            route.model,
            route.provider,
            self.workspace.last_changed_files.clone(),
        );
        self.report.set_outcome(outcome.clone());
        outcome
    }

    /// Take the turn background baseline and kill anything started after it.
    /// Second call is a no-op (baseline already taken).
    fn take_and_kill_turn_backgrounds(&mut self) -> usize {
        match self.workspace.active_turn_background_baseline.take() {
            Some(before) => self.runtime.background().kill_started_after(&before),
            None => 0,
        }
    }

    /// A shared interrupt handle the UI can set to skip the current tool call.
    /// The agent checks it between tool executions; when set, the current tool's
    /// result is replaced with "interrupted by user" and the flag is cleared.
    pub fn interrupt_handle(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.interrupt.clone()
    }

    /// The model id currently configured for this session.
    pub fn model(&self) -> &str {
        &self.config.routing.model
    }

    /// Capture the model and token/window settings so a caller can temporarily
    /// use a different model for one turn and restore the previous route exactly.
    pub fn model_state(&self) -> crate::AgentModelState {
        crate::AgentModelState {
            model: self.config.routing.model.clone(),
            context_window: self.config.routing.context_window,
            requested_max_tokens: self.config.routing.requested_max_tokens,
            max_tokens: self.config.routing.max_tokens,
            max_tokens_explicit: self.config.routing.max_tokens_explicit,
        }
    }

    /// Restore a model state captured by [`Agent::model_state`].
    pub fn restore_model_state(&mut self, state: crate::AgentModelState) {
        self.config.routing.model = state.model;
        self.config.routing.context_window = state.context_window;
        self.config.routing.requested_max_tokens = state.requested_max_tokens;
        self.config.routing.max_tokens = state.max_tokens;
        self.config.routing.max_tokens_explicit = state.max_tokens_explicit;
    }

    /// Switch the model used for subsequent turns, refreshing live metadata
    /// that drives the usage display and output-token budget.
    pub fn set_model(
        &mut self,
        model: String,
        context_window: Option<u32>,
        max_output_tokens: Option<u32>,
    ) {
        self.config.routing.model = model;
        self.config.routing.context_window = context_window;
        self.config.routing.max_tokens = hi_ai::effective_coding_agent_max_tokens(
            &self.config.routing.model,
            self.config.routing.requested_max_tokens,
            self.config.routing.max_tokens_explicit,
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
        self.config.routing.requested_max_tokens = requested_max_tokens;
        self.config.routing.max_tokens_explicit = max_tokens_explicit;
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
        // Static project guides + skills, then live task-ranked memory, then
        // per-turn repo orientation. Memory is separate so it can refresh
        // mid-session without reloading HI.md/skills.
        let combined_context = {
            let mut parts = Vec::new();
            if let Some(project) = self.config.memory.project_context.as_deref() {
                let t = project.trim();
                if !t.is_empty() {
                    parts.push(t.to_string());
                }
            }
            if let Some(mem) = self.task.memory_context.as_deref() {
                let t = mem.trim();
                if !t.is_empty() {
                    parts.push(t.to_string());
                }
            }
            if let Some(task) = self.task.task_context.as_deref() {
                let t = task.trim();
                if !t.is_empty() {
                    parts.push(t.to_string());
                }
            }
            (!parts.is_empty()).then(|| parts.join("\n\n"))
        };
        SystemPrompt::new()
            .with_workspace_root(self.runtime.root())
            .with_project_context(combined_context.as_deref())
            .with_goal(goal)
            .with_goal_state(goal_section.as_deref())
            .with_decisions(decisions.prompt_section().as_deref())
            .with_finalize(self.config.memory.finalize)
            .build()
    }

    /// Reload project + global memory, rank bullets for `task`, and cache the
    /// prompt section. Cheap (two small file reads + sort). Call at turn start
    /// and after coding-fact writes so new bullets land in the next model call.
    pub(crate) fn refresh_memory_context(&mut self, task: &str) {
        let project = crate::memory::read_project_annotated_at(self.runtime.root());
        let global = crate::memory::read_global_memory();
        let next = crate::memory::memory_section_for_task(&project, &global, task);
        if next != self.task.memory_context {
            self.task.set_memory_context(next);
        }
    }

    pub(crate) fn system_message(&self) -> Message {
        self.system_message_for(
            self.goals.free_text.as_deref(),
            self.goals.structured.as_ref(),
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
        self.goals.free_text.as_deref()
    }

    /// The durable intra-session decision log (recorded via `record_decision`),
    /// injected into the system prompt each turn and preserved across compaction.
    pub fn decisions(&self) -> &DecisionLog {
        &self.decisions
    }

    /// Set or clear the transient session goal and inject it into the system prompt.
    pub fn set_goal(&mut self, goal: Option<String>) {
        self.goals.set_free_text(goal);
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
    /// Only takes effect when `config.subagents.long_horizon` is on; when set, the goal's
    /// state is injected into the system prompt each turn so the agent resumes
    /// the active sub-goal. Returns whether it was accepted.
    pub fn set_structured_goal(&mut self, goal: Option<Goal>) -> Result<bool> {
        if !self.config.subagents.long_horizon && goal.is_some() {
            return Ok(false);
        }
        if let Some(session) = self.session.as_mut() {
            if let Some(g) = &goal {
                session.record_goal(g)?;
            } else {
                session.clear_goal()?;
            }
        }
        self.goals.set_structured(goal);
        self.refresh_system_message();
        Ok(true)
    }

    /// The structured long-horizon goal, if any (for persistence/observability).
    pub fn structured_goal(&self) -> Option<&Goal> {
        self.goals.structured.as_ref()
    }

    /// Pause or resume the structured goal without losing progress: a paused goal
    /// is dropped from the system prompt and the driver leaves it alone, but its
    /// sub-goal progress is retained and persisted so `/goal resume` picks up
    /// exactly where it left off. Returns whether there was a goal to update.
    pub fn set_goal_paused(&mut self, paused: bool) -> bool {
        self.set_goal_pause_reason(if paused {
            crate::GoalPauseReason::User
        } else {
            crate::GoalPauseReason::None
        })
    }

    /// Pause/resume with a typed reason (`User`, `Stall`, `Review`, …).
    pub fn set_goal_pause_reason(&mut self, reason: crate::GoalPauseReason) -> bool {
        let snapshot = match self.goals.structured.as_mut() {
            Some(goal) => {
                if matches!(reason, crate::GoalPauseReason::None) {
                    goal.resume();
                } else {
                    goal.pause(reason);
                }
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

    /// Mutate the structured goal and persist (events, edits, etc.).
    pub fn update_structured_goal(&mut self, f: impl FnOnce(&mut Goal)) -> Result<bool> {
        let snapshot = match self.goals.structured.as_mut() {
            Some(goal) => {
                f(goal);
                goal.clone()
            }
            None => return Ok(false),
        };
        if let Some(session) = self.session.as_mut() {
            session.record_goal(&snapshot)?;
        }
        // Keep domain state identical to snapshot (f already mutated in place).
        self.refresh_system_message();
        Ok(true)
    }

    /// Export goal checklist markdown to `.hi/goal-plan.md`.
    pub fn export_goal_plan(&mut self) -> Result<Option<std::path::PathBuf>> {
        let Some(goal) = self.goals.structured.as_ref() else {
            return Ok(None);
        };
        let path = goal.export_markdown_to(self.workspace_root())?;
        let _ = self.update_structured_goal(|g| {
            g.push_event("export", format!("wrote {}", path.display()));
        });
        Ok(Some(path))
    }

    /// Turn the `/goal team` skeptic gate on/off for the active goal. Persists with
    /// the goal (so a resumed goal remembers it) and refreshes the system message.
    /// Returns `false` if there's no active goal.
    pub fn set_goal_team(&mut self, on: bool) -> bool {
        let snapshot = match self.goals.structured.as_mut() {
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
    /// Set (or clear) the goal's drive-turn budget.
    ///
    /// Setting a budget also resets the spend and clears a budget pause, so
    /// `/goal budget 20` after a park means "twenty more turns" rather than
    /// re-parking immediately on the already-spent count.
    pub fn set_goal_turn_budget(&mut self, budget: Option<u32>) -> bool {
        let snapshot = match self.goals.structured.as_mut() {
            Some(goal) => {
                goal.turn_budget = budget;
                // An explicit choice stops the automatic rescaling: from here
                // the number is the user's, and it stays where they put it.
                goal.budget_auto = false;
                goal.turns_spent = 0;
                if goal.pause_reason == crate::goal::GoalPauseReason::Budget {
                    goal.resume();
                }
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

    pub fn set_goal_step_limit(&mut self, limit: Option<usize>) -> bool {
        let snapshot = match self.goals.structured.as_mut() {
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

    /// The per-session turn limit (`/turns`). `None` = unlimited.
    pub fn max_turns(&self) -> Option<u32> {
        self.config.max_turns
    }

    /// How many turns have completed in this session so far.
    pub fn turn_count(&self) -> u32 {
        self.turn_count
    }

    /// Set (or clear, with `None`) the per-session turn limit. Live only — not
    /// persisted with the goal. Takes effect on the next `run_turn` entry.
    pub fn set_max_turns(&mut self, limit: Option<u32>) {
        self.config.max_turns = limit;
    }

    /// One-line goal summary for status surfaces: the structured goal's
    /// progress ("objective — 2/7 sub-goals done", with a paused marker) when one
    /// is set, else the transient goal string, else "off".
    pub fn goal_summary(&self) -> String {
        if let Some(g) = &self.goals.structured {
            let done = g
                .sub_goals
                .iter()
                .filter(|s| s.status == crate::GoalStatus::Done)
                .count();
            let paused = if g.is_paused() {
                format!(" · paused({})", g.pause_reason.as_str())
            } else {
                String::new()
            };
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
            let complete = if g.objective_complete {
                " · objective✓"
            } else {
                ""
            };
            return format!(
                "{} — {}/{} sub-goals done{paused}{skeptic}{complete}",
                g.objective,
                done,
                g.sub_goals.len()
            );
        }
        self.goals
            .free_text
            .clone()
            .unwrap_or_else(|| "off".to_string())
    }

    /// Whether long-horizon agency is on (the `long_horizon` config flag), so
    /// frontends can branch `/goal` between the structured goal and the
    /// transient goal string.
    pub fn long_horizon(&self) -> bool {
        self.config.subagents.long_horizon
    }

    /// Whether a planner model is configured for `/goal` decomposition
    /// ([`decompose_goal`](Self::decompose_goal)).
    pub fn has_planner(&self) -> bool {
        self.config.subagents.planner_model.is_some()
    }

    /// The model the `/goal team` review gate uses: `skeptic_model` when
    /// configured, otherwise the session model. Never empty — the gate works
    /// with zero configuration.
    pub fn effective_skeptic_model(&self) -> &str {
        self.config
            .subagents
            .skeptic_model
            .as_deref()
            .unwrap_or(&self.config.routing.model)
    }

    /// Whether the most recent turn's verification passed (None if not run).
    pub fn last_verify(&self) -> Option<bool> {
        self.report.last_verify
    }

    /// Files whose content or presence changed in the most recent turn.
    pub fn last_changed_files(&self) -> &[String] {
        &self.workspace.last_changed_files
    }

    /// Exact structured file changes reported by tools during the last turn.
    pub fn last_file_changes(&self) -> &[hi_tools::FileChange] {
        &self.workspace.last_file_changes
    }

    /// Merge repeated edits to one path into a turn-level before/after record.
    pub(crate) fn record_tool_effects(&mut self, effects: &hi_tools::ToolEffects) -> Result<()> {
        self.runtime.ledger().record_tool_effects(effects)?;
        if effects.mutation_applied {
            if let Some(contract) = self.task.last_task_contract.as_mut() {
                contract.observe_mutation();
            }
            self.runtime.clear_repo_map_cache();
            self.runtime.invalidate_context();
        }
        self.merge_file_changes(&effects.file_changes);
        Ok(())
    }

    pub(crate) async fn reconcile_workspace_changes(&mut self) -> Result<()> {
        let changes = self.runtime.reconcile_ledger_async().await?;
        if !changes.is_empty() {
            if let Some(contract) = self.task.last_task_contract.as_mut() {
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
                .workspace
                .last_file_changes
                .iter()
                .position(|existing| existing.path == change.path)
            {
                let existing = &self.workspace.last_file_changes[index];
                if existing.before_digest == change.after_digest
                    && existing.before_mode == change.after_mode
                {
                    self.workspace.last_file_changes.remove(index);
                    continue;
                }
                let existing = &mut self.workspace.last_file_changes[index];
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
                self.workspace.last_file_changes.push(change.clone());
            }
        }
    }

    /// Compatibility fallbacks that were triggered in the most recent turn.
    pub fn last_compat_fallbacks(&self) -> &[String] {
        &self.report.last_compat_fallbacks
    }

    /// Telemetry from the most recent turn: verify rounds, recovery retries,
    /// nudges fired, stall flags, and the attributions parsed from the last
    /// verify failure. Lets callers diagnose *how* a turn went, not just
    /// whether it passed.
    pub fn last_turn_telemetry(&self) -> &TurnTelemetry {
        &self.report.last_turn_telemetry
    }

    /// Actual deterministic verification executions retained for the latest
    /// turn, including failed turns that ended during later reconciliation or
    /// provider recovery.
    pub fn last_verification_executions(&self) -> &[crate::VerificationExecution] {
        &self.report.last_turn_telemetry.verification_executions
    }

    /// Typed outcome of the most recent successfully finalized turn.
    pub fn last_turn_outcome(&self) -> Option<&crate::TurnOutcome> {
        self.report.last_turn_outcome.as_ref()
    }

    pub fn last_effective_route(&self) -> &crate::EffectiveModelRoute {
        &self.report.last_effective_route
    }

    /// Provider label supplied by the frontend for the effective route.
    pub fn provider_route(&self) -> Option<&str> {
        self.config.routing.provider_route.as_deref()
    }

    /// The tool mode currently configured for this session.
    pub fn tool_mode(&self) -> ToolMode {
        self.config.routing.tool_mode
    }

    /// A read-only snapshot of all live agent settings for `/config show`.
    pub fn config_snapshot(&self) -> crate::ConfigSnapshot {
        let c = &self.config;
        crate::ConfigSnapshot {
            model: c.routing.model.clone(),
            provider_route: c.routing.provider_route.clone().unwrap_or_default(),
            max_tokens: if c.routing.max_tokens_explicit {
                format!("{} (explicit)", c.routing.max_tokens)
            } else {
                c.routing.max_tokens.to_string()
            },
            thinking_budget: c
                .routing
                .thinking_budget
                .map(|n| n.to_string())
                .unwrap_or_else(|| "off".into()),
            reasoning_effort: c
                .routing
                .reasoning_effort
                .map(|e| e.as_str().to_string())
                .unwrap_or_else(|| "off".into()),
            temperature: c
                .routing
                .temperature
                .map(|t| t.to_string())
                .unwrap_or_else(|| "default".into()),
            max_steps: self.max_steps_setting(),
            tool_mode: c.routing.tool_mode.label().to_string(),
            compat: c.routing.compat.label().to_string(),
            verify: self.verify_summary(),
            review: c.gates.review.label().to_string(),
            lsp: c.gates.lsp_mode.label().to_string(),
            tool_set: c.memory.tool_set.label().to_string(),
            auto_compact: if c.memory.auto_compact {
                format!("on (≥{}%)", c.memory.auto_compact_percent)
            } else {
                "off".into()
            },
            proactive_verify: c.gates.proactive_verify,
            read_only_preflight: c.gates.read_only_preflight,
            long_horizon: c.subagents.long_horizon,
            confirm_edits: c.gates.confirm_edits,
            curate_skills: c.memory.curate_skills,
            explore_subagents: c.subagents.explore_subagents,
            write_subagents: c.subagents.write_subagents.as_str().into(),
            planner_model: c
                .subagents
                .planner_model
                .clone()
                .unwrap_or_else(|| "off".into()),
            skeptic_model: c
                .subagents
                .skeptic_model
                .clone()
                .unwrap_or_else(|| "off".into()),
            moe_streaming: match std::env::var("HI_MLX_EXPERT_STREAMING").as_deref() {
                Ok("0") => "off".into(),
                Ok(_) => "on".into(),
                Err(_) => "auto".into(),
            },
        }
    }

    /// Whether any verification stage is configured.
    pub fn verify_is_on(&self) -> bool {
        !matches!(self.config.gates.verification, VerificationMode::Disabled)
    }

    /// A one-line summary of the verification pipeline (`"off"` when none) —
    /// e.g. `"cargo check → cargo test"`.
    pub fn verify_summary(&self) -> String {
        match &self.config.gates.verification {
            VerificationMode::Disabled => "off".to_string(),
            VerificationMode::Auto => {
                let stages = self
                    .config
                    .gates
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
        &self.config.gates.verification
    }

    /// Stages resolved for the current workspace (empty when disabled or when
    /// automatic detection found no applicable pipeline).
    pub fn resolved_verification_stages(&self) -> Vec<VerifyStage> {
        self.config
            .gates
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
        self.config.gates.verification = verification;
        Ok(())
    }

    /// Replace the verification pipeline (from auto-detection).
    pub fn set_verify_pipeline(&mut self, stages: Vec<VerifyStage>) -> Result<()> {
        let verification = VerificationMode::Explicit(stages);
        verification.validate()?;
        self.config.gates.verification = verification;
        Ok(())
    }

    /// The reasoning effort applied to main-turn requests (`None` = off, i.e. no
    /// `reasoning_effort` sent and the endpoint's own default is used).
    pub fn reasoning_effort(&self) -> Option<hi_ai::ReasoningEffort> {
        self.config.routing.reasoning_effort
    }

    /// Set (or clear, with `None`) the reasoning effort for subsequent turns.
    /// Applies to main-turn requests on OpenAI-compatible endpoints that accept
    /// `reasoning_effort`; the Anthropic adapter and non-supporting endpoints
    /// ignore it. Safe to call between turns (like the other `/`-command setters).
    pub fn set_reasoning_effort(&mut self, effort: Option<hi_ai::ReasoningEffort>) {
        self.config.routing.reasoning_effort = effort;
    }

    /// The sampling temperature applied to requests (`None` = provider default).
    pub fn temperature(&self) -> Option<f32> {
        self.config.routing.temperature
    }

    /// Set (or clear, with `None`) the sampling temperature for subsequent turns.
    pub fn set_temperature(&mut self, temperature: Option<f32>) {
        self.config.routing.temperature = temperature;
    }

    /// Human-readable live step-limit setting. `auto` uses the intent-aware
    /// defaults; `off` uses no practical per-turn cap.
    pub fn max_steps_setting(&self) -> String {
        if !self.config.loop_limits.max_steps_explicit {
            "auto".to_string()
        } else if self.config.loop_limits.max_steps == u32::MAX {
            "off".to_string()
        } else {
            self.config.loop_limits.max_steps.to_string()
        }
    }

    pub fn max_tool_calls_limit(&self) -> u32 {
        self.config.loop_limits.max_tool_calls
    }

    /// Set a fixed per-turn step cap, or disable the cap with `None`.
    pub fn set_max_steps_limit(&mut self, limit: Option<u32>) {
        self.config.loop_limits.max_steps = limit.unwrap_or(u32::MAX).max(1);
        self.config.loop_limits.max_steps_explicit = true;
    }

    /// Restore intent-aware automatic step limits for subsequent turns.
    pub fn set_max_steps_auto(&mut self) {
        self.config.loop_limits.max_steps_explicit = false;
    }

    pub fn rsi_status(&self) -> (&'static str, &'static str, Option<bool>) {
        let requested = if self.config.rsi.enabled { "on" } else { "off" };
        let mode = if self.config.rsi.managed {
            "managed"
        } else if self.config.rsi.enabled {
            "remote"
        } else {
            "off"
        };
        (requested, mode, self.rsi_observe.last_fully_observed)
    }

    pub fn rsi_maximum_cost_microusd(&self) -> Option<u64> {
        self.config
            .rsi
            .control
            .as_ref()
            .map(|control| control.maximum_cost_microusd())
    }

    pub fn rsi_channel(&self) -> &'static str {
        self.config
            .rsi
            .control
            .as_ref()
            .map_or("stable", |control| control.channel())
    }

    pub fn set_rsi_channel(&mut self, channel: crate::command::RsiChannel) -> Result<()> {
        let control = self
            .config
            .rsi
            .control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.set_channel(channel.as_str())
    }

    pub async fn rsi_public_status(&self) -> Result<String> {
        let control = self
            .config
            .rsi
            .control
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
            .rsi
            .control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.set_maximum_cost_microusd(value)
    }

    pub fn set_rsi_enabled(&mut self, enabled: bool) -> Result<()> {
        anyhow::ensure!(
            !self.config.rsi.managed || enabled,
            "managed RSI cannot be disabled"
        );
        if enabled && !self.config.rsi.managed {
            anyhow::ensure!(
                self.config.rsi.remote_switch.is_some(),
                "remote RSI requires PIPENETWORK_API_KEY or an active Pipe provider key"
            );
        }
        self.config.rsi.enabled = enabled;
        if let Some(switch) = &self.config.rsi.remote_switch {
            switch.store(enabled, std::sync::atomic::Ordering::SeqCst);
        }
        Ok(())
    }

    pub async fn set_rsi_enabled_validated(&mut self, enabled: bool) -> Result<()> {
        let control = self.config.rsi.control.clone();
        if enabled && !self.config.rsi.managed {
            let control = control
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
            control.validate().await?;
        }
        if !self.config.rsi.managed
            && let Some(control) = &control
        {
            control.persist_enabled(enabled)?;
        }
        self.set_rsi_enabled(enabled)
    }

    pub async fn rsi_command(&self, argument: &str) -> Result<String> {
        let control = self
            .config
            .rsi
            .control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.command(argument).await
    }

    pub fn set_last_rsi_fully_observed(&mut self, observed: Option<bool>) {
        self.rsi_observe.set_last_fully_observed(observed);
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
            && let Some(goal) = &self.goals.structured
            && let Err(err) = session.record_goal(goal)
        {
            ui.status(&format!("(couldn't persist goal: {err})"));
        }
        // Refresh the human-readable export alongside the durable record.
        // It used to be written only on an explicit `/goal export`, so the file
        // people actually open to check on a long run could sit hours stale
        // while the goal moved underneath it — a supervision surface that
        // silently disagrees with reality is worse than none. Best-effort: a
        // write failure must not disturb a turn that already persisted.
        let root = self.runtime.root().to_path_buf();
        if let Some(goal) = &self.goals.structured {
            let _ = goal.export_markdown_to(&root);
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
