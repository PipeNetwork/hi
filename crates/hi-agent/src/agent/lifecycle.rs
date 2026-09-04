//! `Agent` construction, accessors, and persistence: `new`/`resume`/`undo`,
//! the message/usage/context/goal/verify accessors, system-prompt refresh,
//! and `persist`/`persist_goal`/`messages_mut`.

mod cancellation;
mod commit;
mod drive;
mod goals;
mod mcp;
mod routes;
mod rsi;
mod workspace;
mod workspace_failure;
mod workspace_shutdown;

use std::sync::Arc;

use anyhow::{Context, Result};
use hi_ai::{Message, Provider, Role, ToolMode, Usage, provider_error_usage};

use crate::domain::VerifyEvidence;

use super::tool_selection::{
    BackgroundToolAvailability, advertised_tools, advertised_tools_with_background,
};

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
    pub fn with_extension_registry(
        mut self,
        registry: hi_agent_lifecycle::ExtensionRegistry,
    ) -> Self {
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
            .with_standing_rules(config.memory.standing_rules.as_deref())
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
        mut structured_goal: Option<Goal>,
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
        let migrated_legacy_goal_budget = structured_goal
            .as_mut()
            .is_some_and(crate::Goal::clear_legacy_automatic_budget);
        agent.goals.structured = agent
            .config
            .subagents
            .long_horizon
            .then_some(structured_goal)
            .flatten();
        agent.pending_legacy_goal_budget_migration =
            migrated_legacy_goal_budget && agent.goals.structured.is_some();
        // Seed occupancy so a near-full resume can compact before its first turn.
        agent.report.context_used = crate::compaction::estimate_tokens(agent.messages.as_slice());
        agent.refresh_system_message();
        Ok(agent)
    }

    fn with_messages(
        provider: Arc<dyn Provider>,
        mut config: AgentConfig,
        messages: Vec<Message>,
        persisted: usize,
        scan: Option<crate::change_ledger::BackgroundScan>,
    ) -> Result<Self> {
        let mut messages = Transcript::new(messages);
        // Do not carry an older synthetic FINALIZE_PROMPT into the next turn.
        messages.strip_finalize_pair();
        messages.strip_trailing_nudges();
        messages.strip_previous_turn_blocks();
        messages.repair_for_provider();
        messages
            .validate_and_repair_for_provider()
            .context("loaded transcript is not provider-safe after repair")?;
        // Clamp persisted to the (possibly shorter) transcript length so the
        // incremental session recorder doesn't slice past the end.
        let persisted = persisted.min(messages.len());
        config.gates.verification.validate()?;
        let sandbox_policy = config.sandbox_policy;
        let sandbox_config = config.sandbox_config.clone();
        let initial_lsp_mode = if config.defer_initial_lsp {
            crate::LspMode::Off
        } else {
            config.gates.lsp_mode
        };
        let runtime = WorkspaceRuntime::new_with_scan_sandbox_config_and_project_hooks(
            &config.paths.workspace_root,
            &config.paths.state_root,
            initial_lsp_mode,
            scan,
            sandbox_policy,
            sandbox_config,
            !config.suppress_initial_project_hooks,
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
        let engine_runtime =
            hi_engine_host::EngineRuntime::new(hi_engine_host::ModuleValidationPolicy {
                allow_unsigned: config.engine.allow_unsigned,
                trusted_keys: hi_engine_host::parse_trusted_keys(&config.engine.trusted_key_hex)?,
                max_guest_fuel: config.engine.max_guest_fuel,
                max_guest_memory_bytes: config.engine.max_guest_memory_bytes,
                max_guest_step_ms: config.engine.max_guest_step_ms,
                ..hi_engine_host::ModuleValidationPolicy::default()
            })?;
        if config.engine.mode == hi_engine_api::EngineMode::Wasm {
            if let Some(module_path) =
                hi_engine_host::discover_module_path(config.engine.module_path.as_deref())
            {
                match engine_runtime.reload(&module_path) {
                    Ok(_) => {
                        config.engine.module_path = Some(module_path.clone());
                        if config.engine.watch
                            && let Err(error) = engine_runtime.start_watch(module_path)
                        {
                            tracing::warn!(%error, "WASM engine watch could not start");
                        }
                    }
                    Err(error) => {
                        // The optional logic module must not make ordinary
                        // native turns unusable. The rejection is retained in
                        // logs and the config surface still makes the selected
                        // mode/path visible.
                        tracing::warn!(%error, "WASM engine module rejected; using native engine");
                    }
                }
            } else {
                tracing::warn!(
                    "WASM engine selected but no module was found; using native engine until a validated module is loaded"
                );
            }
        }
        let btw_jobs = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let btw_dispatch = crate::agent::turn::btw::BtwDispatcher::new(btw_jobs.clone());
        let workspace_coordination =
            crate::workspace_coordination::WorkspaceCoordination::new_local_with_settings(
                &config.paths.workspace_root,
                &config.paths.state_root,
                config.harness.clone(),
            );
        let bg_tasks = Arc::new(hi_tools::BackgroundTaskRegistry::new_with_limits(
            hi_tools::BackgroundTaskLimits {
                max_tasks: config.harness.jobs.max_active,
                max_concurrent_preparations: config.harness.jobs.max_preparations,
                queue_timeout: config.harness.jobs.queue_timeout,
            },
        ));
        workspace_coordination.bind_background_registries(runtime.background(), &bg_tasks);
        Ok(Self {
            provider,
            provider_capability_registry: hi_ai::ProviderCapabilityRegistry::default(),
            skeptic_provider,
            local_skeptic: None,
            team_local_servers: Vec::new(),
            driver_local_server: None,
            config,
            engine_runtime,
            side_call_timeout: crate::agent::turn::DEFAULT_SIDE_CALL_TIMEOUT,
            runtime,
            workspace_coordination,
            workspace_durability: None,
            task: crate::domain::TaskContextState::default(),
            messages,
            tools,
            session: None,
            persisted,
            totals: Usage::default(),
            pending_prompt: None,
            usage_pricing: None,
            report: crate::domain::TurnReportState::new(last_effective_route),
            workspace: crate::domain::WorkspaceTurnState::default(),
            subagents: crate::domain::SubagentSessionState::default(),
            bg_tasks,
            interrupt: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            turn_cancellation: None,
            #[cfg(test)]
            undo_test_probe: None,
            repair_effort_escalated: false,
            goals: crate::domain::GoalState::default(),
            pending_legacy_goal_budget_migration: false,
            decisions: DecisionLog::default(),
            snapshot_cache: SnapshotCache::default(),
            prefix_stability: crate::prefix_stability::PrefixStability::default(),
            token_budget: crate::token_budget::TokenBudgetState::default(),
            interjections: crate::InterjectionInbox::default(),
            btw_jobs,
            btw_dispatch,
            btw_git_facts_cache: std::sync::Mutex::new(None),
            pending_block: None,
            rsi_observe: crate::domain::RsiObserveState::default(),
            plan_mode: false,
            plan_drive_pause: crate::plan_drive::PlanDrivePause::Running,
            plan_approval_parked: false,
            plan_drive_stall: 0,
            goal_drive_stall: 0,
            plan_drive_evidence: crate::plan_drive::DriveEvidenceLedger::default(),
            goal_drive_evidence: crate::plan_drive::DriveEvidenceLedger::default(),
            interactive_session: false,
            drive_restore_permission: None,
            goal_requeue_notice: None,
            ask_user_calls: 0,
            ask_user_drive_streak: 0,
            turn_drive_kind: crate::DriveKind::User,
            pending_plan_interruption_resume: false,
            turn_consumed_plan_interruption: false,
            permission_mode: crate::PermissionMode::default(),
            approval_parked: false,
            turn_count: 0,
            last_suggested_prompt: None,
            extensions: None,
            mcp: None,
            memory: None,
        })
    }

    /// Connect markdown memory tools. Advertises inject-gated search/get/update/forget.
    pub fn attach_memory(&mut self, backend: Arc<dyn hi_tools::MemoryBackend>) {
        self.memory = Some(backend);
        self.config.memory.offer_memory = true;
    }

    /// Attach connected MCP servers. Advertises `search_tool` / `use_tool` on
    /// later turns; does not flatten each server tool's schema onto the request.
    pub fn attach_mcp(&mut self, backend: Arc<dyn hi_tools::McpBackend>) {
        self.mcp = Some(backend);
        self.config.memory.offer_mcp = true;
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
        self.undo_with_ledger_reconcile(true).await
    }

    /// Cancellation already performs one bounded ledger reconciliation after
    /// rollback. Deferring this scan keeps an otherwise successful restore from
    /// starting an unbounded cleanup wait before that cancellation backstop.
    pub(crate) async fn undo_without_ledger_reconcile(&mut self) -> Result<Option<usize>> {
        self.undo_with_ledger_reconcile(false).await
    }

    async fn undo_with_ledger_reconcile(
        &mut self,
        reconcile_ledger: bool,
    ) -> Result<Option<usize>> {
        if self.workspace.checkpoints.is_empty() {
            return Ok(None);
        }
        self.begin_durable_workspace_mutation(None).await?;
        let operation = self.undo_inner(reconcile_ledger).await;
        let mut execution = hi_workspace::ExecutionReport {
            disposition: if operation.is_ok() {
                hi_workspace::ExecutionDisposition::Succeeded
            } else {
                hi_workspace::ExecutionDisposition::Failed
            },
            workspace_may_have_changed: true,
            external_effect_may_have_occurred: false,
            content_digest: reconcile_ledger.then(|| self.runtime.ledger().workspace_revision()),
            changed_paths: Vec::new(),
            artifacts: Vec::new(),
            detail: Some(match &operation {
                Ok(Some(restored)) => format!("undo restored {restored} workspace entries"),
                Ok(None) => "undo completed without a restorable checkpoint".into(),
                Err(error) => format!("undo execution failed: {error:#}"),
            }),
        };
        let transcript = [hi_ai::Content::Text(
            "Workspace undo operation completed.".into(),
        )];
        if let Err(error) = self.stage_active_workspace_execution(&[], &transcript, &[], &execution)
        {
            execution.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
            execution.content_digest = None;
            execution.detail = Some(format!(
                "undo transcript staging is ambiguous after execution: {error:#}"
            ));
        }
        let durability = self
            .checkpoint_durable_workspace_with_execution(execution)
            .await;
        match (operation, durability) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error.context(
                "undo changed local bytes but the PipeFS revision was not committed; run /pipefs retry",
            )),
            (Err(error), Err(durability_error)) => Err(error.context(format!(
                "undo failed and PipeFS reconciliation also failed: {durability_error:#}"
            ))),
        }
    }

    async fn undo_inner(&mut self, reconcile_ledger: bool) -> Result<Option<usize>> {
        let Some(reference) = self.workspace.checkpoints.last().cloned() else {
            return Ok(None);
        };
        #[cfg(test)]
        if let Some((delay, calls)) = self.undo_test_probe.clone() {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(delay).await;
        }
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
            let reconcile = if reconcile_ledger {
                self.runtime.reconcile_ledger_async().await.map(|_| ())
            } else {
                Ok(())
            };
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
        if reconcile_ledger {
            self.runtime.reconcile_ledger_async().await?;
        }
        self.runtime.clear_repo_map_cache();
        self.workspace.last_changed_files.clear();
        self.workspace.last_file_changes.clear();
        Ok(Some(n))
    }

    /// Attach a session sink that records messages produced from here on.
    pub fn set_session(&mut self, session: Box<dyn SessionSink>) {
        self.session = Some(session);
        self.publish_model_context();
        if let Err(error) = self.persist_pending_legacy_goal_budget_migration() {
            // `set_session` predates fallible attachment. Keep the migration
            // dirty so the next durable boundary retries it and surfaces an
            // error instead of silently treating the old budget as persisted.
            tracing::warn!(%error, "could not persist migrated legacy goal budget");
        }
    }

    fn persist_pending_legacy_goal_budget_migration(&mut self) -> Result<()> {
        if !self.pending_legacy_goal_budget_migration {
            return Ok(());
        }
        let Some(goal) = self.goals.structured.as_ref() else {
            self.pending_legacy_goal_budget_migration = false;
            return Ok(());
        };
        let Some(session) = self.session.as_mut() else {
            return Ok(());
        };
        session
            .record_goal(goal)
            .context("persisting normalized legacy goal budget")?;
        self.pending_legacy_goal_budget_migration = false;
        Ok(())
    }

    /// Tell the session sink which model this agent runs, so a remote viewer
    /// sees the truth even across `/provider` switches.
    fn publish_model_context(&mut self) {
        let model = self.config.routing.model.clone();
        let window = self.config.routing.context_window;
        if let Some(session) = self.session.as_mut() {
            session.record_model_context(&model, window);
        }
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
        mut structured_goal: Option<Goal>,
        decisions: DecisionLog,
        plan: Vec<hi_tools::PlanStep>,
    ) -> Result<()> {
        let mut messages = crate::Transcript::new(history);
        // Run the same repair pipeline as `with_messages` so a resumed session
        // is cleaned up identically regardless of whether it came from a local
        // JSONL file or remote ipop records.
        messages.strip_finalize_pair();
        messages.strip_trailing_nudges();
        messages.strip_previous_turn_blocks();
        messages.repair_for_provider();
        messages
            .validate_and_repair_for_provider()
            .context("loaded transcript is not provider-safe after repair")?;
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
        let migrated_legacy_goal_budget = structured_goal
            .as_mut()
            .is_some_and(crate::Goal::clear_legacy_automatic_budget);
        self.goals.structured = self
            .config
            .subagents
            .long_horizon
            .then_some(structured_goal)
            .flatten();
        self.pending_legacy_goal_budget_migration =
            migrated_legacy_goal_budget && self.goals.structured.is_some();
        // Clear per-turn / transient state from the previous session, matching
        // what `with_messages` initializes to None/empty for a fresh agent.
        self.goals.free_text = None;
        self.goals.set_plan_if_pending(plan);
        self.workspace.last_changed_files = Vec::new();
        self.report.last_turn_telemetry = TurnTelemetry::default();
        self.report.last_turn_outcome = None;
        self.report.verify = VerifyEvidence::none();
        self.approval_parked = false;
        // Mode and terminal-outcome latches belong to the old session. Restore
        // the normal tool catalog without changing process permission settings;
        // the caller restores the new session's durable approval/drive gates.
        self.set_plan_mode(false);
        // Re-seed the context gauge for the switched-in transcript (see
        // `resume`): carrying the previous session's value either disables
        // graceful compaction or triggers it spuriously.
        self.report.context_used = crate::compaction::estimate_tokens(self.messages.as_slice());
        self.refresh_system_message();
        // The transcript was replaced, so any cached working-tree snapshot is
        // stale. Clear it so the next turn re-snapshots from scratch.
        self.invalidate_snapshot();
        self.runtime.clear_read_cache();
        self.persist_pending_legacy_goal_budget_migration()?;
        Ok(())
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
        runner.set_max_steps(self.max_steps_limit());
        runner.set_max_tool_calls(self.max_tool_calls_cap());
        self.subagents.delegate_runner = Some(runner);
    }

    /// Set the write-capable `delegate` policy at runtime (`/delegate on|off|risk`)
    /// — re-advertises the tool set accordingly. A [`DelegateRunner`] must be
    /// attached for it to actually run.
    pub fn set_write_subagents(&mut self, policy: crate::WriteSubagentPolicy) {
        self.config.subagents.write_subagents = policy;
        self.set_advertised_tools(None);
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
        self.set_advertised_tools(Some((task, intent)));
    }

    fn set_advertised_tools(&mut self, task: Option<(&str, crate::TaskIntent)>) {
        self.tools = self.advertised_tools_for(task);
        if self.tools.iter().any(|tool| tool.name == "new_context") {
            self.token_budget.note_advertised();
        }
    }

    fn advertised_tools_for(
        &self,
        task: Option<(&str, crate::TaskIntent)>,
    ) -> std::sync::Arc<[hi_ai::ToolSpec]> {
        let mut specs: Vec<_> = advertised_tools_with_background(
            &self.config,
            task,
            BackgroundToolAvailability {
                shell: !self.runtime.background().ids().is_empty(),
                tasks: self.bg_tasks.has_tasks(),
                interactive: self.interactive_session,
            },
        )
        .iter()
        .cloned()
        .collect();
        if !self.delegate_runner_matches_workspace() {
            // Never advertise a runner that cannot prove the portable root.
            specs.retain(|spec| spec.name != "delegate");
        }
        // `run_program` is negotiated at the provider boundary rather than
        // inserted into the global catalog. This keeps ordinary providers and
        // text-only routes byte-for-byte on the existing tool set.
        if self.config.program.mode_enabled()
            && self.provider.capabilities().native_tool_calls
            && !matches!(self.config.routing.tool_mode, ToolMode::ChatOnly)
            && !specs.is_empty()
            && !specs.iter().any(|spec| spec.name == "run_program")
        {
            specs.push(hi_tools::run_program_tool_spec());
        }
        if !matches!(self.config.memory.tool_set, crate::ToolSet::Minimal) {
            let window = self.config.routing.context_window.filter(|w| *w > 0);
            let occupancy =
                window.map(|w| crate::token_budget::occupancy_percent(self.report.context_used, w));
            if self.token_budget.should_advertise(
                occupancy,
                self.config.subagents.is_subagent,
                window.is_some(),
            ) && !specs.iter().any(|spec| spec.name == "new_context")
            {
                specs.push(hi_tools::new_context_tool_spec());
            }
        }
        specs.into()
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
        self.restore_state_snapshot_with_workspace_rollback(snapshot, false);
    }

    pub(crate) fn restore_state_snapshot_with_workspace_rollback(
        &mut self,
        snapshot: &crate::AgentStateSnapshot,
        workspace_rolled_back: bool,
    ) {
        // Keep checklist progress from the abandoned turn (see prefer_plan_progress).
        let plan = if workspace_rolled_back {
            crate::domain::GoalState::prefer_plan_progress_after_workspace_rollback(
                &snapshot.last_plan,
                self.goals.plan(),
            )
        } else {
            crate::domain::GoalState::prefer_plan_progress(&snapshot.last_plan, self.goals.plan())
        };
        let plan_drive_scope_changed = crate::heuristics::next_plan_step_title(&snapshot.last_plan)
            != crate::heuristics::next_plan_step_title(&plan);
        self.goals.restore_triple(
            snapshot.goal.clone(),
            snapshot.structured_goal.clone(),
            plan,
        );
        if plan_drive_scope_changed {
            self.plan_drive_stall = 0;
            self.plan_drive_evidence.clear();
        }
        self.decisions = snapshot.decisions.clone();
        self.refresh_system_message();
    }

    /// Durably discard a turn by rewinding both the transcript and the
    /// prompt-injected side state to a pre-turn snapshot.
    ///
    /// Plan checklist progress is normally retained: an interrupt after a
    /// bookkeeping-only `update_plan` should not show a stale checklist. The
    /// cancellation owner uses the rollback-aware variant below when it also
    /// restored workspace files, because completion claims backed by those
    /// removed effects are no longer true.
    pub fn rewind_to_snapshot_durable(
        &mut self,
        len: usize,
        snapshot: &crate::AgentStateSnapshot,
    ) -> Result<()> {
        self.rewind_to_snapshot_durable_with_workspace_rollback(len, snapshot, false)
    }

    pub(crate) fn rewind_to_snapshot_durable_with_workspace_rollback(
        &mut self,
        len: usize,
        snapshot: &crate::AgentStateSnapshot,
        workspace_rolled_back: bool,
    ) -> Result<()> {
        let len = len.min(self.messages.len());
        let mut next = self.messages.as_slice()[..len].to_vec();
        let structured_goal = self
            .config
            .subagents
            .long_horizon
            .then_some(snapshot.structured_goal.clone())
            .flatten();
        let plan = if workspace_rolled_back {
            crate::domain::GoalState::prefer_plan_progress_after_workspace_rollback(
                &snapshot.last_plan,
                self.goals.plan(),
            )
        } else {
            crate::domain::GoalState::prefer_plan_progress(&snapshot.last_plan, self.goals.plan())
        };
        let plan_drive_scope_changed = crate::heuristics::next_plan_step_title(&snapshot.last_plan)
            != crate::heuristics::next_plan_step_title(&plan);
        // Durable session: keep unfinished progress; drop a fully-done checklist
        // so resume does not resurrect it (live UI still shows finished below).
        let session_plan: &[hi_tools::PlanStep] =
            if crate::heuristics::plan_has_pending_steps(&plan) {
                plan.as_slice()
            } else {
                &[]
            };
        // The stable system message carries no goal/decision state — the
        // restored snapshot state below reaches the model via the next
        // turn's volatile context block.
        let system = self.system_message_for();
        if let Some(first) = next.first_mut() {
            *first = system;
        } else {
            next.push(system);
        }

        // Session rewinds must serialize the durable latch, not the effective
        // UI state. During a transactional user resume the badge is hidden,
        // but a crash before successful settlement must still restore paused.
        let plan_drive_paused = self.durable_plan_drive_paused();
        let plan_drive_resume_on_user_input = self.plan_drive_resumes_on_user_input();
        if let Some(session) = self.session.as_mut() {
            if plan_drive_scope_changed {
                // Reset the old scope first. A crash/failure between these two
                // append-only records then leaves the old plan with a clean
                // ledger, never the new next step with inherited stall/evidence.
                session.record_plan_drive_state_with_policy(
                    plan_drive_paused,
                    0,
                    plan_drive_resume_on_user_input,
                    true,
                    &[],
                )?;
            }
            session.record_state_replacement(
                &next,
                structured_goal.as_ref(),
                &snapshot.decisions,
                session_plan,
            )?;
        }
        self.messages.replace_all(next);
        self.persisted = self.messages.len();
        self.goals
            .restore_triple(snapshot.goal.clone(), structured_goal, plan);
        if plan_drive_scope_changed {
            self.plan_drive_stall = 0;
            self.plan_drive_evidence.clear();
        }
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

    /// Install optional provider pricing metadata for normalized telemetry.
    /// Passing `None` keeps usage counts but marks cost as unavailable.
    pub fn set_usage_pricing(&mut self, pricing: Option<(f64, f64)>) {
        self.usage_pricing = pricing;
    }

    /// Live input/output USD-per-1M-token rates, when the catalog published them.
    pub fn usage_pricing(&self) -> Option<(f64, f64)> {
        self.usage_pricing
    }

    /// Run a typed prompt. Image parts are preserved in the provider-neutral
    /// transcript while the existing text-oriented turn machinery continues
    /// to supply task contracts, verification, and guardrails.
    pub async fn run_prompt(
        &mut self,
        input: hi_ai::PromptInput,
        ui: &mut dyn Ui,
    ) -> Result<crate::TurnOutcome> {
        let text = input.text_content();
        self.pending_prompt = Some(input);
        let result = self.run_turn(&text, ui).await;
        // Do not let an early turn-limit/cancellation path leak a typed prompt
        // into the following ordinary string turn.
        self.pending_prompt = None;
        result
    }

    /// Provider/model-independent usage record for the most recent turn.
    pub fn last_usage_telemetry(&self) -> hi_ai::NormalizedUsage {
        hi_ai::NormalizedUsage::new(
            self.report.last_effective_route.provider.clone(),
            self.config.routing.provider_route.clone(),
            self.report.last_effective_route.model.clone(),
            self.report.last_turn_usage,
            self.usage_pricing,
        )
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
                "context: {} / {} tokens ({}% used, window {})\n",
                humanize_count(self.report.context_used),
                humanize_count(u64::from(w)),
                pct,
                self.token_budget.window_id,
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
                    "  ⚠ at or past the auto-compact threshold — /compact to reclaim now, or /window to drop history without a summary\n",
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
        out.push_str("  /window: drop conversation, keep goal/decisions and the current task\n");
        out.push('\n');
        out.push_str(&self.context_injection_census());
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
        ui.session_usage(&self.totals);
        ui.rate_limits(self.totals.rate_limits);
    }

    /// The model id currently configured for this session.
    pub fn model(&self) -> &str {
        &self.config.routing.model
    }

    /// The execution mode currently applied to subsequent turns.
    pub fn execution_mode(&self) -> crate::ExecutionMode {
        self.config.execution
    }

    /// Change the execution mode for subsequent turns.
    ///
    /// Durable execution is deliberately fail-closed: enabling it without a
    /// session sink would make the TUI claim resumability that it cannot
    /// provide. The current transcript is checkpointed before the mode change
    /// is reported as active.
    pub fn set_execution_mode(&mut self, mode: crate::ExecutionMode) -> anyhow::Result<()> {
        if mode == self.config.execution {
            return Ok(());
        }
        if mode.is_durable() && self.session.is_none() {
            anyhow::bail!(
                "durable execution requires a saved session; start without --no-save or attach a SessionSink"
            );
        }

        let previous = self.config.execution;
        self.config.execution = mode;
        if mode.is_durable()
            && let Err(error) = self.persist()
        {
            self.config.execution = previous;
            return Err(error.context("could not checkpoint before enabling durable execution"));
        }
        Ok(())
    }

    /// Change the tool execution mode for subsequent turns.
    pub fn set_tool_mode(&mut self, mode: hi_ai::ToolMode) {
        self.config.routing.tool_mode = mode;
        self.refresh_tools_for_task("", crate::TaskIntent::ReadOnly);
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
        self.publish_model_context();
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
        self.publish_model_context();
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
        self.token_budget = crate::token_budget::TokenBudgetState::default();
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

    /// The **stable** system message: identity/rules, working directory, and
    /// the durable project guides (HI.md/skills). Everything that changes
    /// turn-to-turn lives in [`Self::volatile_context_block`] instead, so
    /// message[0] stays byte-stable and provider prompt caches keep hitting
    /// across a session's many model rounds.
    fn system_message_for(&self) -> Message {
        SystemPrompt::new()
            .with_workspace_root(self.runtime.root())
            .with_project_context(self.config.memory.project_context.as_deref())
            .with_standing_rules(self.config.memory.standing_rules.as_deref())
            .with_finalize(self.config.memory.finalize)
            .build()
    }

    /// The per-turn volatile context: task-ranked memory, the task context
    /// index / repo orientation, sanitized git identity (branch/HEAD/origin
    /// when this workspace *is* the git toplevel), the session goal,
    /// long-horizon goal state, the decision log, a matching stack skill or
    /// the code-review pack on review-shaped turns, and named acceptance
    /// criteria. Attached to each turn's user message (late
    /// in the transcript) rather than the system message — rebuilding
    /// message[0] with this content every round invalidated the entire prefix
    /// for implicit and explicit prompt caches alike (observed: <4% cache
    /// reads on an edit-heavy session). Mid-turn staleness is fine: each
    /// source only changes through the model's own actions (its edits, its
    /// `update_plan`/`record_decision` calls), which it already sees.
    pub(crate) fn volatile_context_block(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
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
        if let Some(goal) = self.goals.free_text.as_deref() {
            let t = goal.trim();
            if !t.is_empty() {
                parts.push(format!("[Current session goal]\n{t}"));
            }
        }
        if let Some(section) = self
            .goals
            .structured
            .as_ref()
            .and_then(|g| g.prompt_section())
        {
            let t = section.trim();
            if !t.is_empty() {
                parts.push(t.to_string());
            }
        }
        if let Some(section) = self.decisions.prompt_section() {
            let t = section.trim();
            if !t.is_empty() {
                parts.push(t.to_string());
            }
        }
        if let Some(section) = crate::git_identity::prompt_section(self.runtime.root()) {
            parts.push(section);
        }
        if !self.config.subagents.is_subagent {
            parts.push(crate::today::prompt_section());
        }
        let tool_free_response = self
            .task
            .last_task_prompt
            .as_deref()
            .is_some_and(crate::task_contract::prompt_requests_tool_free_response);
        if !tool_free_response && !self.config.subagents.is_subagent && self.turn_prompt_is_review()
        {
            if self.config.memory.inject_review_skill
                && let Some(section) = crate::skills::active_review_skill_section()
            {
                parts.push(section);
            }
        } else if !tool_free_response
            && self.config.memory.inject_stack_skill
            && let Some(section) = crate::skills::active_stack_skill_section(self.runtime.root())
        {
            parts.push(section);
        }
        if let Some(section) = self
            .task
            .last_task_contract
            .as_ref()
            .and_then(|c| c.acceptance_section())
        {
            parts.push(section);
        }
        if !self.config.subagents.is_subagent && self.config.memory.offer_ask_user {
            parts.push(
                "ask_user is for product/design forks only — never instead of the next coding step."
                    .into(),
            );
        }
        if let Some(budget) = self.token_budget.fragment() {
            parts.push(budget.to_string());
        }
        // This block carries canonical task/goal requirements as well as
        // summaries. Keep it intact here; `ensure_request_fits_context` owns
        // explicit model-window fitting and can compact with full knowledge of
        // the request instead of silently discarding the tail up front.
        (!parts.is_empty()).then(|| parts.join("\n\n"))
    }

    /// Whether the current turn's stored prompt is a read-only review.
    /// Uses the same classifiers as the turn loop so `/review`, `/security`,
    /// and implicit "review the codebase" match.
    fn turn_prompt_is_review(&self) -> bool {
        // Planning is its own read-only scope. Treating it as code review
        // injects defect-first review instructions and answer-repair gates into
        // a turn whose successful output is an unfinished implementation plan.
        if self.plan_mode {
            return false;
        }
        let Some(prompt) = self.task.last_task_prompt.as_deref() else {
            return false;
        };
        if crate::task_contract::prompt_requests_tool_free_response(prompt) {
            return false;
        }
        if crate::steering::classify_read_only_intent(prompt).is_some() {
            return true;
        }
        let read_only = self
            .task
            .last_task_contract
            .as_ref()
            .is_some_and(|contract| contract.intent == crate::TaskIntent::ReadOnly);
        crate::steering::implicit_read_only_review_intent(prompt, read_only).is_some()
    }

    /// Reload project + global memory, rank bullets for `task`, and cache the
    /// prompt section. Cheap (two small file reads + sort). Call at turn start
    /// and after coding-fact writes so new bullets land in the next model call.
    pub(crate) fn refresh_memory_context(&mut self, task: &str) {
        let project = crate::memory::read_project_annotated_at(self.runtime.root());
        let global = crate::memory::read_global_memory();
        let mut next = crate::memory::memory_section_for_task(&project, &global, task);
        // Findings-ledger steering: when this project's recent turns keep
        // dying the same way, say so up front so the model adapts (e.g. runs
        // the package-local check itself when verification keeps failing).
        // The targeted shape is remembered so findings recorded under the
        // hint carry it — that recurrence data is how a hint earns its keep.
        let hint = self
            .config
            .memory
            .learning
            .then(|| crate::learning::context_hint(self.runtime.state_root()))
            .flatten();
        self.task.active_hint_shape = hint.as_ref().map(|h| h.shape.clone());
        if let Some(hint) = hint {
            next = Some(match next {
                Some(section) => format!("{section}\n{}", hint.text),
                None => hint.text,
            });
        }
        if next != self.task.memory_context {
            self.task.set_memory_context(next);
        }
    }

    pub(crate) fn system_message(&self) -> Message {
        self.system_message_for()
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

    /// Replace message[0] only when the stable system content actually
    /// changed. Callers fire this liberally (goal updates, memory writes,
    /// every turn start); since the volatile context moved out of the system
    /// message this is almost always a no-op — which is the point: an
    /// unchanged message[0] keeps the request prefix byte-stable so provider
    /// prompt caches hit.
    pub(crate) fn refresh_system_message(&mut self) {
        let system = self.system_message();
        self.messages.replace_system_if_changed(system);
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
        if self.skeptic_route_is_dead() {
            return &self.config.routing.model;
        }
        self.config
            .subagents
            .skeptic_model
            .as_deref()
            .unwrap_or(&self.config.routing.model)
    }

    /// True when completion review / goal skeptic share the session model —
    /// i.e. the "second model" gate is not actually independent. Surfaces on
    /// [`crate::TurnOutcome::review_same_model`].
    pub fn skeptic_shares_session_model(&self) -> bool {
        self.effective_skeptic_model() == self.config.routing.model
    }

    /// Whether the most recent turn's verification passed (None if not run).
    pub fn last_verify(&self) -> Option<bool> {
        self.report.verify.as_bool()
    }

    /// Files whose content or presence changed in the most recent turn.
    pub fn last_changed_files(&self) -> &[String] {
        &self.workspace.last_changed_files
    }

    /// Every path the change ledger recorded this session (`/files`, `/commit`).
    pub fn session_touched_paths(&self) -> Vec<String> {
        self.runtime.ledger().touched_paths_since(0)
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
            // Typed mutations already tell us the cache is stale. Clear it
            // eagerly so the next orientation lookup does not walk and stat
            // the entire workspace just to rediscover that fact.
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
            // External edits are discovered by the ledger, so invalidate the
            // task index at the same boundary instead of fingerprinting the
            // whole workspace on every repo_map/find_symbol call.
            self.runtime.clear_repo_map_cache();
            // The read cache is deliberately per-turn, but editors and other
            // processes can still change files between two tool calls. A
            // ledger-observed external change must evict cached file content
            // before the next read or the model can inspect stale text.
            self.runtime.clear_read_cache();
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

    /// Lifetime raw change-ledger events compacted into exact aggregate state.
    pub fn ledger_events_dropped(&self) -> u64 {
        self.runtime.ledger().dropped_event_count()
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
            execution: c.execution.as_str().into(),
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
            top_p: c
                .routing
                .top_p
                .map(|p| p.to_string())
                .unwrap_or_else(|| "default".into()),
            output_token_parameter: c.routing.output_token_parameter.label().to_string(),
            max_steps: self.max_steps_setting(),
            max_tool_calls: self.max_tool_calls_setting(),
            tool_mode: c.routing.tool_mode.label().to_string(),
            compat: c.routing.compat.label().to_string(),
            deepseek_compat: c.routing.deepseek_compat.label().to_string(),
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
            suggest_next_prompt: c.memory.suggest_next_prompt,
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
            engine_mode: c.engine.mode.as_str().to_string(),
            engine_module: c
                .engine
                .module_path
                .as_deref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<none>".into()),
        }
    }

    /// Inspect or reload the optional decision engine without touching the
    /// active turn. Reloads are generation-pinned by `EngineRuntime` and are
    /// therefore safe to request while a provider stream is still running.
    pub fn engine_command(&mut self, argument: &str) -> String {
        let mut parts = argument.split_whitespace();
        let action = parts.next().unwrap_or("status").to_ascii_lowercase();
        match action.as_str() {
            "status" | "show" => {
                let status = hi_engine_host::status(&self.engine_runtime);
                let current = status.current.map_or_else(
                    || "none".to_string(),
                    |module| {
                        format!(
                            "v{} (generation {}, {})",
                            module.guest_version,
                            module.generation,
                            &module.module_sha256[..12]
                        )
                    },
                );
                let pending = status.pending.map_or_else(
                    || "none".to_string(),
                    |module| {
                        format!(
                            "v{} (generation {})",
                            module.guest_version, module.generation
                        )
                    },
                );
                format!(
                    "engine: {}\n  module: {}\n  current: {current}\n  pending: {pending}\n  watch: {}",
                    self.config.engine.mode.as_str(),
                    self.config
                        .engine
                        .module_path
                        .as_deref()
                        .map_or("<auto>", |path| path.to_str().unwrap_or("<non-utf8>")),
                    if self.engine_runtime.is_watching() {
                        "on"
                    } else {
                        "off"
                    },
                )
            }
            "native" | "rust" | "off" => {
                self.config.engine.mode = hi_engine_api::EngineMode::Native;
                self.engine_runtime.stop_watch();
                "engine mode: native (WASM remains loaded but is not selected)".into()
            }
            "wasm" | "component" => {
                self.config.engine.mode = hi_engine_api::EngineMode::Wasm;
                if let Some(path) = parts.next().map(std::path::PathBuf::from) {
                    self.config.engine.module_path = Some(path);
                }
                self.reload_engine_module()
            }
            "reload" => self.reload_engine_module(),
            "watch" => match parts.next().unwrap_or("on") {
                "off" | "disable" => {
                    self.config.engine.watch = false;
                    self.engine_runtime.stop_watch();
                    "engine module watch: off".into()
                }
                "on" | "enable" => {
                    let Some(path) = self.engine_module_path() else {
                        return "engine watch unavailable: set HI_ENGINE_MODULE or provide /engine wasm <path>".into();
                    };
                    match self.engine_runtime.start_watch(path) {
                        Ok(()) => {
                            self.config.engine.watch = true;
                            "engine module watch: on (reloads become active next turn)".into()
                        }
                        Err(error) => format!("engine watch failed: {error:#}"),
                    }
                }
                other => format!("usage: /engine watch [on|off] — got {other:?}"),
            },
            other => format!(
                "usage: /engine [status|native|wasm [path]|reload|watch on|off] — got {other:?}"
            ),
        }
    }

    fn engine_module_path(&self) -> Option<std::path::PathBuf> {
        hi_engine_host::discover_module_path(self.config.engine.module_path.as_deref())
    }

    fn reload_engine_module(&mut self) -> String {
        let Some(path) = self.engine_module_path() else {
            return "engine reload unavailable: set HI_ENGINE_MODULE or place engine.wasm beside hi".into();
        };
        match self.engine_runtime.reload(&path) {
            Ok(info) => {
                self.config.engine.module_path = Some(path.clone());
                format!(
                    "engine candidate loaded: v{} generation {} — active next turn ({})",
                    info.guest_version,
                    info.generation,
                    path.display()
                )
            }
            Err(error) => format!("engine reload rejected: {error:#}"),
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

    /// Human-readable live step-limit setting. `off` means model rounds are
    /// unlimited, which is also the ordinary default.
    pub fn max_steps_setting(&self) -> String {
        if self.config.loop_limits.max_steps == u32::MAX {
            "off".to_string()
        } else {
            self.config.loop_limits.max_steps.to_string()
        }
    }

    /// Numeric model-round cap for child-process launchers. `None` means the
    /// ordinary unlimited/default behavior.
    pub fn max_steps_limit(&self) -> Option<u32> {
        (self.config.loop_limits.max_steps != u32::MAX).then_some(self.config.loop_limits.max_steps)
    }

    /// Human-readable live tool-call limit. `off` means ordinary turns have no
    /// count ceiling; managed workers and explicit CLI caps render their finite
    /// effective value.
    pub fn max_tool_calls_setting(&self) -> String {
        if self.config.loop_limits.max_tool_calls == u32::MAX {
            "off".to_string()
        } else {
            self.config.loop_limits.max_tool_calls.to_string()
        }
    }

    pub fn max_tool_calls_limit(&self) -> u32 {
        self.config.loop_limits.max_tool_calls
    }

    /// Numeric tool-call cap for child-process launchers. `None` means the
    /// ordinary unlimited/default behavior.
    pub fn max_tool_calls_cap(&self) -> Option<u32> {
        (self.config.loop_limits.max_tool_calls != u32::MAX)
            .then_some(self.config.loop_limits.max_tool_calls)
    }

    /// Numeric verification-repair cap for child-process launchers. `None`
    /// means the ordinary unlimited/default behavior.
    pub fn max_verify_repairs_cap(&self) -> Option<u32> {
        (self.config.gates.max_verify_repairs != crate::UNLIMITED_REPAIR_CYCLES)
            .then_some(self.config.gates.max_verify_repairs)
    }

    /// Set a fixed per-turn step cap, or disable the cap with `None`.
    pub fn set_max_steps_limit(&mut self, limit: Option<u32>) {
        self.config.loop_limits.max_steps = limit.unwrap_or(u32::MAX).max(1);
        if let Some(runner) = &self.subagents.delegate_runner {
            runner.set_max_steps(self.max_steps_limit());
        }
    }

    /// Restore the unlimited automatic per-turn model-round default.
    pub fn set_max_steps_auto(&mut self) {
        self.config.loop_limits.max_steps = crate::MAX_MODEL_ROUNDS;
        if let Some(runner) = &self.subagents.delegate_runner {
            runner.set_max_steps(None);
        }
    }

    pub(crate) fn persist(&mut self) -> Result<()> {
        self.persist_pending_legacy_goal_budget_migration()?;
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

    /// Persist the current transcript at a safe execution boundary. Durable
    /// mode deliberately fails the turn when its checkpoint cannot be written:
    /// continuing after a lost checkpoint would make the advertised recovery
    /// guarantee false.
    pub(crate) fn persist_durable_boundary(&mut self, boundary: &str) -> Result<()> {
        if self.config.execution.is_durable() {
            self.persist().with_context(|| {
                format!("durable execution checkpoint failed at {boundary} boundary")
            })?;
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
        //
        // Skip when the workspace root is the process cwd: that is the bare
        // default in canned-provider tests (which run with the crate dir as the
        // root), and exporting there leaks a stub `.hi/goal-plan.md` into the
        // package source tree on every test run. Real sessions and
        // IsolatedWorkspace tests set an explicit root, so they still export.
        let root = self.runtime.root().to_path_buf();
        let is_cwd_default = std::env::current_dir()
            .ok()
            .and_then(|cwd| cwd.canonicalize().ok())
            .and_then(|cwd| {
                root.canonicalize()
                    .ok()
                    .map(|canonical_root| canonical_root == cwd)
            })
            .unwrap_or(false);
        // Goal snapshots are UI/runtime state, not user workspace content.
        // They cannot use the asynchronous PipeFS durability fence from this
        // synchronous state update, so never emit them into a portable root.
        if !self.pipefs_workspace_active()
            && !is_cwd_default
            && let Some(goal) = &self.goals.structured
        {
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

impl Drop for crate::Agent {
    fn drop(&mut self) {
        // Frontends may temporarily retain an Arc clone so they can cancel a
        // task while a turn borrows the Agent. Do not let that observer handle
        // extend the lifetime of child agents after the owning Agent is gone.
        self.bg_tasks.shutdown();
    }
}
