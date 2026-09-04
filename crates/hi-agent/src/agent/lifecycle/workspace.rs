//! Workspace checkpoints, abnormal-turn cleanup, and resumable snapshots.

use std::sync::Arc;

use anyhow::{Context, Result};
use hi_ai::Provider;

use crate::AgentConfig;

/// Retain deterministic evidence across a later, unrelated turn failure only
/// when a pass is still bound to the current workspace. Provider/session
/// failures are not verifier infrastructure failures, and therefore default to
/// `Unverified` rather than manufacturing `InfrastructureError`.
fn verification_after_turn_failure(
    evidence: &crate::domain::VerifyEvidence,
    workspace_reconciled: bool,
    current_revision: u64,
    current_digest: &str,
) -> (crate::VerificationStatus, Option<String>) {
    if !workspace_reconciled {
        return (crate::VerificationStatus::Unverified, None);
    }
    match evidence {
        crate::domain::VerifyEvidence::Passed { revision, digest }
            if *revision == current_revision && digest == current_digest =>
        {
            (crate::VerificationStatus::Passed, Some(digest.clone()))
        }
        crate::domain::VerifyEvidence::Failed => (crate::VerificationStatus::Failed, None),
        crate::domain::VerifyEvidence::None | crate::domain::VerifyEvidence::Passed { .. } => {
            (crate::VerificationStatus::Unverified, None)
        }
    }
}

impl crate::Agent {
    /// Install or clear the host-provided durability fence used by PipeFS.
    pub fn set_workspace_durability(
        &mut self,
        durability: Option<Arc<dyn crate::WorkspaceDurability>>,
    ) {
        self.workspace_durability = durability;
        // Portable workspaces currently deny write-capable child agents: the
        // frontend delegate runner captures a concrete root, and background
        // writers cannot participate in the parent's durability fence. Refresh
        // the advertised set immediately so `delegate` is not offered there.
        self.set_advertised_tools(None);
    }

    pub fn workspace_durability_enabled(&self) -> bool {
        self.workspace_durability.is_some()
    }

    /// Replace repository-supplied prompt context after a controlled root
    /// switch. Standing user rules remain session-scoped; only files from the
    /// newly materialized workspace are re-read by the frontend.
    pub fn set_workspace_project_context(&mut self, context: Option<String>) {
        self.config.memory.project_context = context;
        self.refresh_system_message();
    }

    pub async fn begin_durable_workspace_mutation(
        &self,
        dirty_paths: Option<Vec<String>>,
    ) -> Result<()> {
        if let Some(durability) = &self.workspace_durability {
            durability.mutation_started(dirty_paths).await?;
        }
        Ok(())
    }

    pub async fn checkpoint_durable_workspace(&self) -> Result<()> {
        if let Some(durability) = &self.workspace_durability {
            durability.checkpoint().await?;
        }
        Ok(())
    }

    /// Recreate all workspace-scoped runtime state against a materialized root.
    /// Switching is only legal while idle; callers perform remote preparation
    /// before invoking this method and retain the old runtime on any failure.
    pub async fn rebind_workspace(
        &mut self,
        workspace_root: impl AsRef<std::path::Path>,
        state_root: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        self.rebind_workspace_with_project_hooks(workspace_root, state_root, true)
            .await
    }

    /// Rebind to an untrusted portable materialization without loading any
    /// repository-provided hooks. The caller may still reconnect explicitly
    /// trusted, non-workspace services after the switch.
    pub async fn rebind_portable_workspace(
        &mut self,
        workspace_root: impl AsRef<std::path::Path>,
        state_root: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        self.rebind_workspace_with_project_hooks(workspace_root, state_root, false)
            .await
    }

    /// Complete a launch-root runtime whose executable integrations were
    /// deferred until PipeFS authority was known.  Unlike a rebind this keeps
    /// the agent's restored task state, workspace checkpoints, and transcript
    /// bookkeeping intact.
    pub fn activate_deferred_local_workspace_runtime(&mut self) {
        self.runtime
            .activate_trusted_local_integrations(self.config.gates.lsp_mode);
    }

    async fn rebind_workspace_with_project_hooks(
        &mut self,
        workspace_root: impl AsRef<std::path::Path>,
        state_root: impl AsRef<std::path::Path>,
        allow_project_hooks: bool,
    ) -> Result<()> {
        self.runtime
            .background()
            .ensure_quiescent_and_reaped()
            .await
            .context("cannot switch workspaces before background processes are fully stopped")?;
        let background_tasks = self.bg_tasks.active_ids().await;
        anyhow::ensure!(
            background_tasks.is_empty(),
            "cannot switch workspaces while background tasks remain active: {}",
            background_tasks.join(", ")
        );

        #[cfg(test)]
        let sandbox_policy = self.config.sandbox_policy;
        #[cfg(not(test))]
        let sandbox_policy = None;
        let replacement = crate::WorkspaceRuntime::new_with_scan_sandbox_and_project_hooks(
            workspace_root.as_ref(),
            state_root.as_ref(),
            self.config.gates.lsp_mode,
            None,
            sandbox_policy,
            allow_project_hooks,
        )?;
        // Do not disable the old runtime until every fallible replacement
        // construction step has succeeded. This keeps a failed rebind atomic
        // from the caller's perspective.
        self.runtime.lsp().set_enabled(false).await;
        self.config.paths.workspace_root = replacement.root().to_path_buf();
        self.config.paths.state_root = replacement.state_root().to_path_buf();
        self.runtime = replacement;
        if self.memory.is_some() {
            self.memory = Some(Arc::new(crate::MarkdownMemory::new(
                self.config.paths.workspace_root.clone(),
                true,
            )));
        }
        // Workspace MCP processes and imported repository configuration were
        // resolved for the old root. Do not carry those executable connections
        // across a trust boundary; the frontend may reconnect them explicitly.
        self.mcp = None;
        self.config.memory.offer_mcp = false;
        self.task = crate::domain::TaskContextState::default();
        self.workspace = crate::domain::WorkspaceTurnState::default();
        self.snapshot_cache = crate::snapshot::SnapshotCache::default();
        self.prefix_stability = crate::prefix_stability::PrefixStability::default();
        *self
            .btw_git_facts_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        self.set_advertised_tools(None);
        self.refresh_system_message();
        Ok(())
    }

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

    /// Background handles whose native process is still live. Completed
    /// handles remain queryable for output, but must not block workspace
    /// switching or a final PipeFS checkpoint.
    pub fn active_background_process_ids(&self) -> Vec<String> {
        self.runtime
            .background()
            .snapshot()
            .into_iter()
            .filter(|(_, _, status)| status == "running")
            .map(|(id, _, _)| id)
            .collect()
    }

    /// Cloneable native-process registry for host durability controllers. The
    /// handle is workspace-runtime scoped and is acquired only after a PipeFS
    /// rebind, so lease loss can terminate processes still mutating that cache.
    pub fn background_process_registry(&self) -> Arc<hi_tools::BackgroundRegistry> {
        self.runtime.background_arc()
    }

    /// Require all native background processes to be terminal and fully reaped.
    /// A process marked killed can still execute its final write until its
    /// detached driver has observed exit, so lifecycle callers must await this
    /// barrier before removing a PipeFS materialization.
    pub fn ensure_background_processes_quiescent(
        &self,
    ) -> impl std::future::Future<Output = Result<()>> + Send + 'static {
        let background = self.runtime.background_arc();
        async move { background.ensure_quiescent_and_reaped().await }
    }

    pub async fn observe_durable_background_process(
        &self,
        outcome: &hi_tools::BackgroundOutcome,
    ) -> Result<()> {
        if let Some(durability) = &self.workspace_durability {
            let running = matches!(
                outcome.state,
                hi_tools::BackgroundState::Started | hi_tools::BackgroundState::Running
            );
            durability
                .background_process_state(&outcome.id, running)
                .await?;
        }
        Ok(())
    }

    /// Registry ids for in-process background subagent tasks (`task` tool).
    pub fn background_task_ids(&self) -> Vec<String> {
        self.bg_tasks.list_now()
    }

    /// In-process subagent tasks that have not reached a terminal outcome.
    /// This async snapshot is authoritative and must be used for lifecycle
    /// decisions; `background_task_ids` intentionally includes retained history.
    pub fn active_background_task_ids(
        &self,
    ) -> impl std::future::Future<Output = Vec<String>> + Send + 'static {
        let tasks = Arc::clone(&self.bg_tasks);
        async move { tasks.active_ids().await }
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
                        checkpoint_refs_before,
                    } => {
                        if let Err(err) =
                            self.rollback_turn_checkpoint(&checkpoint_refs_before).await
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

    /// Restore the checkpoint created by the active turn, if one is still on
    /// the stack, then put the exact pre-turn bounded undo history back. A
    /// length-only comparison is insufficient at [`crate::MAX_CHECKPOINTS`]:
    /// adding the new checkpoint evicts the oldest and keeps the same length.
    pub(crate) async fn rollback_turn_checkpoint(
        &mut self,
        checkpoint_refs_before: &[String],
    ) -> Result<usize> {
        let current = &self.workspace.checkpoints;
        let new_checkpoint_is_live =
            current != checkpoint_refs_before && current.len() >= checkpoint_refs_before.len();
        if !new_checkpoint_is_live {
            return Ok(0);
        }

        let restored_files = self.undo().await?.unwrap_or(0);
        if self.workspace.checkpoints == checkpoint_refs_before {
            return Ok(restored_files);
        }

        // A full stack evicted its oldest reference when the active checkpoint
        // was appended. `undo` removes the active tail; restore that evicted
        // reference as well so cancellation is state-neutral.
        if let Some(session) = self.session.as_mut() {
            session.record_checkpoints(checkpoint_refs_before)?;
        }
        self.workspace.checkpoints = checkpoint_refs_before.to_vec();
        Ok(restored_files)
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
        let workspace_reconciled = self.runtime.ledger().reconcile().is_ok();
        let (current_revision, current_digest, changes) = {
            let mut ledger = self.runtime.ledger();
            (
                ledger.revision(),
                ledger.workspace_revision(),
                ledger.changes_since(baseline),
            )
        };
        let (verification, verified_workspace_revision) = verification_after_turn_failure(
            &self.report.verify,
            workspace_reconciled,
            current_revision,
            &current_digest,
        );
        self.workspace.record_changes(changes, true);
        self.report.clear_verify();
        self.workspace.clear_active_baselines();
        let route = self.report.last_effective_route.clone();
        let mut outcome = crate::TurnOutcome::infrastructure_failure(
            route.model,
            route.provider,
            self.workspace.last_changed_files.clone(),
        );
        outcome.verification = verification;
        outcome.verified_workspace_revision = verified_workspace_revision;
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
            plan_drive_evidence: self.plan_drive_evidence.snapshot(),
            goal_drive_evidence: self.goal_drive_evidence.snapshot(),
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
        agent
            .plan_drive_evidence
            .restore(snapshot.plan_drive_evidence);
        agent
            .goal_drive_evidence
            .restore(snapshot.goal_drive_evidence);
        Ok(agent)
    }
}

#[cfg(test)]
mod failure_attribution_tests {
    use super::verification_after_turn_failure;
    use crate::VerificationStatus;
    use crate::domain::VerifyEvidence;

    #[test]
    fn only_a_current_revision_pass_survives_later_infrastructure_failure() {
        let pass = VerifyEvidence::pass(7, "current".into());
        assert_eq!(
            verification_after_turn_failure(&pass, true, 7, "current"),
            (VerificationStatus::Passed, Some("current".into()))
        );
        assert_eq!(
            verification_after_turn_failure(&pass, true, 8, "changed"),
            (VerificationStatus::Unverified, None)
        );
        assert_eq!(
            verification_after_turn_failure(&pass, false, 7, "current"),
            (VerificationStatus::Unverified, None)
        );
    }
}
