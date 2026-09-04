//! Workspace checkpoints, abnormal-turn cleanup, and resumable snapshots.

use std::sync::Arc;

use anyhow::{Context, Result};
use hi_ai::{Content, Provider};

use super::workspace_failure::verification_after_turn_failure;
use crate::AgentConfig;

impl crate::Agent {
    /// Install or clear the host-provided durability fence used by PipeFS.
    pub fn set_workspace_durability(
        &mut self,
        durability: Option<Arc<dyn crate::WorkspaceDurability>>,
    ) {
        self.workspace_durability = durability;
        if self.workspace_durability.is_none()
            && let Err(error) = self.workspace_coordination.install_local(
                &self.config.paths.workspace_root,
                &self.config.paths.state_root,
            )
        {
            tracing::error!(%error, "could not restore the local workspace controller");
        }
        // Refresh tools because portable workspaces deny write-capable delegates.
        self.set_advertised_tools(None);
    }

    pub fn workspace_durability_enabled(&self) -> bool {
        self.workspace_durability.is_some()
    }

    /// Whether the active workspace authority is PipeFS. Policy gates must use
    /// this binding fact rather than the optional legacy durability adapter.
    pub fn pipefs_workspace_active(&self) -> bool {
        matches!(
            self.workspace_controller_binding().authority,
            hi_workspace::WorkspaceAuthority::PipeFs { .. }
        )
    }

    pub fn workspace_controller_binding(&self) -> hi_workspace::WorkspaceBinding {
        self.workspace_coordination.binding()
    }

    pub fn workspace_controller_capabilities(&self) -> hi_workspace::WorkspaceCapabilities {
        self.workspace_coordination.capabilities()
    }

    pub fn workspace_controller_status(&self) -> hi_workspace::WorkspaceStatus {
        self.workspace_coordination.status()
    }

    pub fn harness_settings(&self) -> &hi_workspace::ResolvedHarnessSettings {
        &self.config.harness
    }

    pub fn harness_session_layer(&self) -> Option<&hi_workspace::SettingLayer> {
        self.config.harness_session.as_ref()
    }

    pub fn activate_pipefs_workspace_controller(
        &self,
        session_id: &str,
        writer_protocol: u16,
        causal_commit: bool,
    ) -> Result<()> {
        self.workspace_coordination.install_pipefs(
            session_id,
            writer_protocol,
            causal_commit && self.config.harness.features.pipefs_causal_commit_v1,
            &self.config.paths.workspace_root,
            &self.config.paths.state_root,
        )
    }

    /// Install a controller that owns its backend settlement path.
    pub fn install_workspace_controller(
        &self,
        controller: Arc<dyn hi_workspace::WorkspaceController>,
    ) -> Result<()> {
        self.workspace_coordination.install_controller(controller)
    }

    pub async fn acknowledge_workspace_recovery(&mut self) -> Result<()> {
        self.workspace_coordination
            .reconcile_after_external_proof()
            .await
    }

    /// Persist remote authority so resume still probes IPOP after cache cleanup.
    pub fn record_pipefs_mode(&mut self, enabled: bool) -> Result<()> {
        if let Some(session) = self.session.as_mut() {
            session.record_pipefs_mode(enabled)?;
        }
        Ok(())
    }

    /// Replace repository-supplied prompt context after a controlled root switch.
    pub fn set_workspace_project_context(&mut self, context: Option<String>) {
        self.config.memory.project_context = context;
        self.refresh_system_message();
    }

    pub async fn begin_durable_workspace_mutation(
        &self,
        dirty_paths: Option<Vec<String>>,
    ) -> Result<()> {
        self.workspace_coordination
            .begin(self.workspace_durability.clone(), dirty_paths)
            .await
    }

    pub(crate) async fn begin_classified_workspace_operation(
        &self,
        intent: hi_workspace::MutationIntent,
    ) -> Result<()> {
        self.workspace_coordination
            .begin_intent(self.workspace_durability.clone(), intent)
            .await?;
        if let Err(error) =
            hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::ToolBeforeStart)
        {
            self.workspace_coordination.abandon_active()?;
            return Err(error.into());
        }
        Ok(())
    }

    /// Stage the exact provider-facing result before a PipeFS settlement.
    /// Local workspaces publish their transcript after the local checkpoint and
    /// therefore need no pre-stage. Remote workspaces must place this record in
    /// the durable outbox first so the controller cannot commit bytes against a
    /// one-step-behind transcript batch.
    pub(crate) fn stage_active_workspace_execution(
        &mut self,
        calls: &[(String, String, String)],
        assistant_content: &[Content],
        results: &[(String, String)],
        execution: &hi_workspace::ExecutionReport,
    ) -> Result<()> {
        if !matches!(
            self.workspace_controller_binding().authority,
            hi_workspace::WorkspaceAuthority::PipeFs { .. }
        ) {
            return Ok(());
        }
        let operation_id = match self.workspace_coordination.active_parent_operation() {
            Some(operation_id) => operation_id,
            None if !self.config.harness.features.workspace_controller_v2 => return Ok(()),
            None => anyhow::bail!("PipeFS execution has no admitted workspace operation"),
        };
        anyhow::ensure!(
            calls.len() == results.len(),
            "workspace execution transcript has {} calls but {} results",
            calls.len(),
            results.len()
        );
        let transcript_calls = calls
            .iter()
            .zip(results)
            .map(|((call_id, name, _), (result_id, result))| {
                anyhow::ensure!(
                    call_id == result_id,
                    "workspace execution result order does not match call order"
                );
                Ok(crate::WorkspaceTranscriptCall {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    result: result.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let record = crate::WorkspaceTranscriptExecution {
            schema_version: crate::WorkspaceTranscriptExecution::SCHEMA_VERSION,
            operation_id,
            assistant_content: assistant_content.to_vec(),
            calls: transcript_calls,
            execution: execution.clone(),
        };
        self.session
            .as_mut()
            .context("PipeFS execution requires a durable session sink")?
            .stage_workspace_execution(&record)
            .context("staging PipeFS workspace execution transcript")
    }

    /// Settle an admitted operation using the executor's real typed result.
    /// Storage success must never rewrite a failed/cancelled/indeterminate
    /// execution into `Succeeded` in the operation journal.
    pub(crate) async fn checkpoint_durable_workspace_with_execution(
        &self,
        mut execution: hi_workspace::ExecutionReport,
    ) -> Result<()> {
        if execution.workspace_may_have_changed && execution.content_digest.is_none() {
            execution.content_digest = Some(self.runtime.ledger().workspace_revision());
        }
        let pending = self.runtime.background().pending_job_settlements().await;
        self.workspace_coordination
            .checkpoint(self.workspace_durability.clone(), execution)
            .await?;
        self.runtime
            .background()
            .settle_jobs_after_workspace(&pending)
            .await
    }

    /// Recreate all workspace-scoped runtime state against a materialized root.
    /// Switching is only legal while idle; callers perform remote preparation
    /// before invoking this method and retain the old runtime on any failure.
    pub async fn rebind_workspace(
        &mut self,
        workspace_root: impl AsRef<std::path::Path>,
        state_root: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        self.rebind_workspace_with_project_hooks(workspace_root, state_root, true, true)
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
        self.rebind_workspace_with_project_hooks(workspace_root, state_root, false, true)
            .await
    }

    /// Persist the last-write-wins checkpoint boundary which separates two
    /// concrete workspace roots. Hosts which require a remote durability
    /// barrier before changing roots can record and flush this boundary first,
    /// then call [`Self::rebind_workspace_after_durable_boundary`].
    pub fn record_workspace_checkpoint_boundary(&mut self) -> Result<()> {
        if let Some(session) = self.session.as_mut() {
            session
                .record_checkpoints(&[])
                .context("persisting the workspace checkpoint-generation boundary")?;
        }
        Ok(())
    }

    /// Rebind after [`Self::record_workspace_checkpoint_boundary`] has already
    /// been durably flushed. This avoids creating an unacknowledged transcript
    /// record after a remote workspace has been disabled.
    pub async fn rebind_workspace_after_durable_boundary(
        &mut self,
        workspace_root: impl AsRef<std::path::Path>,
        state_root: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        self.rebind_workspace_with_project_hooks(workspace_root, state_root, true, false)
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
        record_checkpoint_boundary: bool,
    ) -> Result<()> {
        // Hold the exclusive admission side from the first stable drain
        // through controller publication. Job terminal callbacks remain
        // available, but no cloned registry can cross the final barrier on
        // the old binding.
        let rebind_admission = self
            .workspace_coordination
            .close_admission_for_rebind()
            .await;
        self.ensure_workspace_rebind_ready()?;
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
        self.require_workspace_barrier(hi_workspace::BarrierKind::Rebind)
            .await
            .context("waiting for the unified workspace rebind barrier")?;
        hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::RebindAfterDrain)
            .map_err(anyhow::Error::from)?;

        let sandbox_policy = self.config.sandbox_policy;
        let sandbox_config = self.config.sandbox_config.clone();
        let replacement = crate::WorkspaceRuntime::new_with_scan_sandbox_config_and_project_hooks(
            workspace_root.as_ref(),
            state_root.as_ref(),
            self.config.gates.lsp_mode,
            None,
            sandbox_policy,
            sandbox_config,
            allow_project_hooks,
        )?;
        // Keep the old runtime until fallible replacement construction succeeds.
        // from the caller's perspective.
        // Checkpoint references address snapshots of one concrete workspace.
        // Append an explicit empty, last-write-wins boundary before switching
        // roots so a crash or later resume can never apply an old root's undo
        // snapshot to the newly activated workspace.
        if record_checkpoint_boundary {
            self.record_workspace_checkpoint_boundary()?;
        }
        self.workspace_coordination.install_local_during_rebind(
            replacement.root(),
            replacement.state_root(),
            &rebind_admission,
        )?;
        self.runtime.lsp().set_enabled(false).await;
        self.config.paths.workspace_root = replacement.root().to_path_buf();
        self.config.paths.state_root = replacement.state_root().to_path_buf();
        self.runtime = replacement;
        self.workspace_coordination
            .bind_background_registries(self.runtime.background(), &self.bg_tasks);
        self.bind_delegate_runner_workspace();
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
        drop(rebind_admission);
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

    /// Legacy synchronous cancelled-turn finalizer.
    ///
    /// Use [`Self::cleanup_turn`] so background kill and bounded ledger
    /// reconciliation stay consistent across frontends. Call
    /// [`Self::finalize_cancelled_turn_snapshot_only`] only when a deliberately
    /// incomplete, nonblocking fallback is required.
    #[deprecated(
        since = "0.3.1",
        note = "use async Agent::cleanup_turn; use finalize_cancelled_turn_snapshot_only only for an incomplete nonblocking fallback"
    )]
    pub fn finalize_cancelled_turn(&mut self) -> Result<crate::TurnOutcome> {
        let _ = self.take_and_kill_turn_backgrounds();
        // Preserve the historical public API's ordering and full, blocking
        // reconciliation for downstream callers while steering new async
        // frontends to `cleanup_turn`. Truncate before the fallible scan just
        // as the pre-deprecation implementation did.
        if let Some(start) = self.workspace.active_turn_message_start.take() {
            self.truncate_messages(start);
        }
        let explicit_baseline = self.workspace.active_turn_ledger_revision;
        let changes = {
            let mut ledger = self.runtime.ledger();
            ledger.reconcile()?;
            let baseline = explicit_baseline.unwrap_or_else(|| ledger.revision());
            ledger.changes_since(baseline)
        };
        self.workspace.active_turn_ledger_revision = None;
        self.finalize_cancelled_turn_with_changes(changes)
    }

    /// Finalize a cancelled turn using only changes already present in the
    /// ledger. This never starts or waits for a workspace scan, so unobserved
    /// shell/editor effects may be absent from `TurnOutcome::changed_files`.
    pub fn finalize_cancelled_turn_snapshot_only(&mut self) -> Result<crate::TurnOutcome> {
        let _ = self.take_and_kill_turn_backgrounds();
        self.finalize_cancelled_turn_inner()
    }

    /// Legacy synchronous failed-turn finalizer.
    ///
    /// Use [`Self::cleanup_turn`] with [`TurnCleanupKind::Fail`], which performs
    /// a bounded scan for otherwise-unobserved shell/editor effects. Call
    /// [`Self::finalize_failed_turn_snapshot_only`] only when a deliberately
    /// incomplete, nonblocking fallback is required.
    #[deprecated(
        since = "0.3.1",
        note = "use async Agent::cleanup_turn; use finalize_failed_turn_snapshot_only only for an incomplete nonblocking fallback"
    )]
    pub fn finalize_failed_turn(&mut self) -> crate::TurnOutcome {
        let _ = self.take_and_kill_turn_backgrounds();
        // Compatibility wrapper: the legacy method attempted a synchronous
        // full scan and ignored its error. It also selected the fallback
        // baseline before that scan, so callers without an explicit active
        // turn still observed the resulting delta. Keep both behaviors for
        // external callers; in-tree async paths use bounded cleanup instead.
        let explicit_baseline = self.workspace.active_turn_ledger_revision.take();
        let (changes, current_workspace, workspace_reconciled) = {
            let mut ledger = self.runtime.ledger();
            let baseline = explicit_baseline.unwrap_or_else(|| ledger.revision());
            let workspace_reconciled = ledger.reconcile().is_ok();
            (
                ledger.changes_since(baseline),
                Some((ledger.revision(), ledger.workspace_revision())),
                workspace_reconciled,
            )
        };
        self.finalize_failed_turn_with_changes(changes, workspace_reconciled, current_workspace)
    }

    /// Finalize a failed turn using only changes already present in the ledger.
    /// This never starts or waits for a workspace scan or a busy ledger mutex,
    /// so unobserved shell/editor effects may be absent from
    /// `TurnOutcome::changed_files`.
    pub fn finalize_failed_turn_snapshot_only(&mut self) -> crate::TurnOutcome {
        let _ = self.take_and_kill_turn_backgrounds();
        self.finalize_failed_turn_inner(false)
    }

    fn finalize_cancelled_turn_inner(&mut self) -> Result<crate::TurnOutcome> {
        // Message truncate only if still set (AlreadyApplied path takes it first).
        if let Some(start) = self.workspace.active_turn_message_start.take() {
            self.truncate_messages(start);
        }
        let changes = self.take_abnormal_turn_ledger_changes();
        self.finalize_cancelled_turn_with_changes(changes)
    }

    pub(super) fn finalize_cancelled_turn_with_changes(
        &mut self,
        changes: Vec<hi_tools::FileChange>,
    ) -> Result<crate::TurnOutcome> {
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

    fn finalize_failed_turn_inner(&mut self, workspace_reconciled: bool) -> crate::TurnOutcome {
        let (changes, current_workspace) = self.take_abnormal_turn_ledger_snapshot();
        self.finalize_failed_turn_with_changes(changes, workspace_reconciled, current_workspace)
    }

    pub(super) fn finalize_failed_turn_with_changes(
        &mut self,
        changes: Vec<hi_tools::FileChange>,
        workspace_reconciled: bool,
        current_workspace: Option<(u64, String)>,
    ) -> crate::TurnOutcome {
        let (verification, verified_workspace_revision) = current_workspace
            .map(|(current_revision, current_digest)| {
                verification_after_turn_failure(
                    &self.report.verify,
                    workspace_reconciled,
                    current_revision,
                    &current_digest,
                )
            })
            .unwrap_or((crate::VerificationStatus::Unverified, None));
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

    /// Reconcile surviving shell/editor changes without allowing abnormal-turn
    /// cleanup to inherit an unbounded filesystem walk. Dropping the timed
    /// future signals the blocking worker; the short follow-up wait lets it
    /// release the ledger mutex but never takes that mutex on this async task.
    pub(super) async fn reconcile_abnormal_turn_bounded(&self) -> bool {
        const RECONCILE_GRACE: std::time::Duration = std::time::Duration::from_secs(1);
        const RELEASE_GRACE: std::time::Duration = std::time::Duration::from_millis(100);
        if self.workspace.active_turn_ledger_revision.is_none() {
            // Setup can be cancelled while its initial reconciliation owns the
            // ledger, before a turn baseline exists. Dropping that reconcile
            // future signals its blocking worker, but does not join it. Give
            // the worker the same bounded release window used below so the
            // completed cancellation does not leave the next turn racing a
            // still-unwinding setup scan.
            let _ = self.runtime.wait_for_ledger_available(RELEASE_GRACE).await;
            return false;
        }
        match tokio::time::timeout(RECONCILE_GRACE, self.runtime.reconcile_ledger_async()).await {
            Ok(Ok(_)) => return true,
            Ok(Err(error)) => {
                eprintln!("hi-agent: abnormal-turn ledger reconciliation failed: {error:#}");
            }
            Err(_) => {
                eprintln!(
                    "hi-agent: abnormal-turn ledger reconciliation exceeded its {}ms cleanup budget; changed-files reporting may be incomplete",
                    RECONCILE_GRACE.as_millis()
                );
            }
        }
        let _ = self.runtime.wait_for_ledger_available(RELEASE_GRACE).await;
        false
    }

    fn take_abnormal_turn_ledger_changes(&mut self) -> Vec<hi_tools::FileChange> {
        self.take_abnormal_turn_ledger_snapshot().0
    }

    pub(super) fn take_abnormal_turn_ledger_snapshot(
        &mut self,
    ) -> (Vec<hi_tools::FileChange>, Option<(u64, String)>) {
        let baseline = self.workspace.active_turn_ledger_revision.take();
        let Some(mut ledger) = self.runtime.try_ledger() else {
            eprintln!(
                "hi-agent: ledger remained busy after abnormal-turn cleanup; changed-files reporting may be incomplete"
            );
            // `last_file_changes` belongs to the last settled turn until the
            // new prompt reaches its durable boundary. Reusing it here can
            // falsely attribute that prior turn's files to an early-cancelled
            // or failed turn, so an unavailable ledger must fail empty.
            return (Vec::new(), None);
        };
        let baseline = baseline.unwrap_or_else(|| ledger.revision());
        (
            ledger.changes_since(baseline),
            Some((ledger.revision(), ledger.workspace_revision())),
        )
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
#[path = "workspace_failure_tests.rs"]
mod failure_attribution_tests;

#[path = "workspace_checkpoint.rs"]
mod checkpoint;
