//! Workspace checkpoints, abnormal-turn cleanup, and resumable snapshots.

use std::sync::Arc;

use anyhow::Result;
use hi_ai::Provider;

use crate::AgentConfig;

impl crate::Agent {
    /// Number of git checkpoints created so far (for `/undo`).
    pub fn checkpoint_count(&self) -> usize {
        self.workspace.checkpoints.len()
    }

    /// Durable checkpoint references for the current session, newest last.
    pub fn checkpoint_refs(&self) -> &[String] {
        &self.workspace.checkpoints
    }

    /// Explicit root owned by this agent's workspace runtime.
    pub fn workspace_root(&self) -> &std::path::Path {
        self.runtime.root()
    }

    /// Per-project runtime state owned by Hi, outside the user's workspace.
    /// Frontends use this for UI metadata that must not appear as project work.
    pub fn state_root(&self) -> &std::path::Path {
        self.runtime.state_root()
    }

    /// Snapshot this agent runtime's background handles for cancellable turns.
    pub fn background_process_ids(&self) -> Vec<String> {
        self.runtime.background().ids()
    }

    /// Registry ids for in-process background subagent tasks (`task` tool).
    pub fn background_task_ids(&self) -> Vec<String> {
        self.bg_tasks.list_now()
    }

    /// Cloneable handle to the background-task registry. Frontends use this to
    /// cancel a `task` while a turn future still borrows the agent.
    pub fn background_task_registry(&self) -> Arc<hi_tools::BackgroundTaskRegistry> {
        Arc::clone(&self.bg_tasks)
    }

    /// A read-only snapshot of this session's background jobs `(id, command,
    /// status)` — used by the `/btw` session snapshot so the model can answer
    /// "what jobs are running / did my task finish" without polling.
    pub(crate) fn background_snapshot(&self) -> Vec<(String, String, String)> {
        self.runtime.background().snapshot()
    }

    /// Kill only **auto-backgrounded** processes (foreground commands that
    /// outgrew their timeout) started after `before`. Deliberate
    /// `run_in_background` jobs are spared — they are long-lived work that
    /// must survive turn end, cancel, and retry rewinds; they still die with
    /// the session or an explicit `bash_kill`.
    pub fn kill_background_processes_started_after(&self, before: &[String]) -> usize {
        self.runtime.background().kill_started_after(before)
    }

    /// Kill turn-scoped background processes started after this turn's
    /// baseline, without running the full failed-turn finalizer. Used on the
    /// `run_turn_cancellable` error path so a mid-turn provider/tool failure
    /// does not leak delegate/explore subagents started this turn, while
    /// leaving ledger reconciliation and `last_changed_files` to the caller's
    /// own `cleanup_turn(Fail)` / `finalize_failed_turn` (idempotent via
    /// `.take()` on the baseline).
    pub fn kill_turn_backgrounds(&mut self) -> usize {
        self.take_and_kill_turn_backgrounds()
    }

    /// Stop every background process owned by this agent runtime, plus any
    /// auto-managed local skeptic server and team-role model servers, on
    /// session shutdown.
    pub fn kill_background_processes(&self) {
        self.runtime.background().kill_all();
        self.stop_local_skeptic_server();
        if let Some(server) = &self.driver_local_server {
            hi_tools::stop_local_server(&server.process_id);
        }
        for server in &self.team_local_servers {
            hi_tools::stop_local_server(&server.process_id);
        }
        // Background subagent tasks are cleaned up via BackgroundTaskRegistry's
        // Drop impl when the agent is dropped. The async `kill_all` method can
        // be called from async cleanup paths if needed.
    }

    /// Shutdown variant for runs whose deliverable may be a *running service*:
    /// stop hi's own infrastructure (skeptic/team model servers) and any
    /// auto-backgrounded strays, but leave processes the model deliberately
    /// started with `run_in_background: true` alive after exit.
    ///
    /// Observed motivation: one-shot prompts like "set up a server on port
    /// 8080 and keep it running" had their finished deliverable reaped by
    /// `kill_background_processes` microseconds before the caller connected.
    pub fn release_background_services(&self) {
        self.runtime.background().kill_auto_backgrounded();
        self.runtime.background().release_all();
        self.stop_local_skeptic_server();
        if let Some(server) = &self.driver_local_server {
            hi_tools::stop_local_server(&server.process_id);
        }
        for server in &self.team_local_servers {
            hi_tools::stop_local_server(&server.process_id);
        }
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
        // Abort immediate `/btw` so a cancelled/failed turn can't keep answering.
        self.disarm_btw_dispatcher();
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
            review_same_model: self.skeptic_shares_session_model(),
            leftover: None,
            plan_leftover: None,
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
        let mut outcome = crate::TurnOutcome::infrastructure_failure(
            route.model,
            route.provider,
            self.workspace.last_changed_files.clone(),
        );
        outcome.review_same_model = self.skeptic_shares_session_model();
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

    /// Capture enough durable state to resume this conversation in another
    /// frontend process. Callers must snapshot only between turns.
    pub fn session_snapshot(&self) -> crate::AgentSessionSnapshot {
        crate::AgentSessionSnapshot {
            messages: self.messages.as_slice().to_vec(),
            usage: self.totals,
            checkpoint_refs: self.workspace.checkpoints.clone(),
            structured_goal: self.goals.clone_structured(),
            decisions: self.decisions.clone(),
            plan: self.goals.plan().to_vec(),
        }
    }

    /// Resume from a snapshot produced by [`Self::session_snapshot`].
    pub fn resume_snapshot(
        provider: Arc<dyn Provider>,
        config: AgentConfig,
        snapshot: crate::AgentSessionSnapshot,
    ) -> Result<Self> {
        let mut agent = Self::resume(
            provider,
            config,
            snapshot.messages,
            snapshot.usage,
            snapshot.checkpoint_refs,
            snapshot.structured_goal,
            snapshot.decisions,
        )?;
        agent.goals.set_plan_if_pending(snapshot.plan);
        Ok(agent)
    }
}
