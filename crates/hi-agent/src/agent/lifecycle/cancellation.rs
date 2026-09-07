//! Ordered abnormal-turn process, rollback, and workspace settlement barriers.

use anyhow::{Context, Result};
use hi_ai::Content;

impl crate::Agent {
    /// Stop every turn-scoped writer which can still race a cancellation
    /// rollback, and wait for its native child/lifecycle callback to finish.
    /// Deliberate `run_in_background:true` services are not turn-scoped and are
    /// intentionally preserved.
    #[cfg(test)]
    pub(crate) async fn quiesce_abnormal_turn_processes(&self) -> Result<usize> {
        self.quiesce_abnormal_turn_processes_before(
            tokio::time::Instant::now() + std::time::Duration::from_secs(60),
        )
        .await
    }

    pub(crate) async fn quiesce_abnormal_turn_processes_before(
        &self,
        deadline: tokio::time::Instant,
    ) -> Result<usize> {
        let foreground = self.foreground_process_registry();
        foreground.kill_current();

        let baseline = self.workspace.active_turn_background_baseline.clone();
        let signalled = baseline
            .as_deref()
            .map(|before| self.runtime.background().kill_started_after(before))
            .unwrap_or(0);
        let background = self.runtime.background_arc();
        let wait_background = async move {
            match baseline {
                Some(before) => {
                    background
                        .kill_started_after_and_reap_before(&before, deadline)
                        .await
                }
                None => Ok(0),
            }
        };
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let task_baseline = self
            .workspace
            .active_turn_task_baseline
            .as_deref()
            .unwrap_or(&[]);
        let (foreground_reaped, background_reaped, tasks, publications_settled) = tokio::join!(
            foreground.wait_until_empty(remaining),
            wait_background,
            self.bg_tasks
                .kill_started_after_before(task_baseline, deadline),
            self.bg_tasks.wait_for_candidate_publications(remaining),
        );
        anyhow::ensure!(
            foreground_reaped,
            "timed out waiting for a cancelled foreground process to be reaped"
        );
        background_reaped.context("reaping auto-backgrounded turn processes")?;
        anyhow::ensure!(
            publications_settled && tasks.iter().all(|task| task.state.is_terminal()),
            "timed out waiting for candidate publication rollback or recovery settlement"
        );
        Ok(signalled)
    }

    async fn settle_abnormal_workspace_operation(
        &mut self,
        disposition: hi_workspace::ExecutionDisposition,
        workspace_reconciled: bool,
        changes: &[hi_tools::FileChange],
        current_workspace: Option<(u64, String)>,
        allow_preexisting_fence: bool,
    ) -> Result<()> {
        // Reaping runs each process's exactly-once terminal callback. Freeze
        // the resulting durability-pending job set before publishing the
        // workspace receipt, then advance those jobs only after that receipt.
        let pending_jobs = self.runtime.background().pending_job_settlements().await;
        let Some(intent) = self.workspace_coordination.active_intent() else {
            // Workspace admission can fail before this turn owns a permit.
            // The existing controller fence (including another live writer or
            // durable recovery evidence) is not ours to settle or clear.
            if allow_preexisting_fence {
                return Ok(());
            }
            // A tool batch can hand its permit to the shielded settlement task
            // immediately before the outer cancellation branch wins. Do not
            // publish a terminal turn result until that task has left the
            // transient Mutating/Settling states.
            self.workspace_coordination
                .await_known_settlement(std::time::Duration::from_secs(5))
                .await?;
            let status = self.workspace_controller_status();
            anyhow::ensure!(
                matches!(
                    status.state,
                    hi_workspace::WorkspaceState::Ready
                        | hi_workspace::WorkspaceState::LocalAuditDegraded
                ),
                "abnormal turn workspace settlement requires recovery ({:?}): {}",
                status.state,
                status.detail.as_deref().unwrap_or("no detail")
            );
            return self
                .runtime
                .background()
                .settle_jobs_after_workspace(&pending_jobs)
                .await;
        };
        let known = workspace_reconciled && current_workspace.is_some();
        let mut execution = hi_workspace::ExecutionReport {
            disposition: if known {
                disposition
            } else {
                hi_workspace::ExecutionDisposition::Indeterminate
            },
            workspace_may_have_changed: matches!(
                intent.effect_scope,
                hi_workspace::EffectScope::LiveWriter
            ),
            external_effect_may_have_occurred: !matches!(
                intent.replay_class,
                hi_workspace::ReplayClass::PureWorkspace
            ),
            content_digest: current_workspace.map(|(_, digest)| digest),
            changed_paths: changes
                .iter()
                .map(|change| std::path::PathBuf::from(&change.path))
                .collect(),
            artifacts: Vec::new(),
            detail: Some(if known {
                match disposition {
                    hi_workspace::ExecutionDisposition::Cancelled => {
                        "turn cancellation was reaped and reconciled before settlement".into()
                    }
                    _ => "abnormal turn exit was reaped and reconciled before settlement".into(),
                }
            } else {
                "abnormal turn cleanup could not prove the final workspace image".into()
            }),
        };

        let cancellation_record = [Content::Text(
            "Workspace operation ended during abnormal-turn cleanup.".into(),
        )];
        if let Err(error) = self
            .stage_active_workspace_execution(&[], &cancellation_record, &[], &execution)
            .await
        {
            execution.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
            execution.detail = Some(format!(
                "workspace execution transcript could not be staged during cleanup: {error:#}"
            ));
        }

        self.workspace_coordination
            .settle_active(self.workspace_durability.clone(), execution)
            .await?;
        self.workspace_coordination
            .await_known_settlement(std::time::Duration::from_secs(5))
            .await?;
        self.runtime
            .background()
            .settle_jobs_after_workspace(&pending_jobs)
            .await
    }

    /// Single entry point for abnormal turn teardown (cancel / infrastructure fail).
    ///
    /// Owns turn-scoped background kill via the active-turn baseline. Normal
    /// successful turns clear baselines inside `run_turn` and must not call this.
    pub async fn cleanup_turn(
        &mut self,
        kind: crate::TurnCleanupKind,
    ) -> Result<crate::TurnCleanupResult> {
        let deadline = self.turn_cancellation.as_ref().map_or_else(
            || tokio::time::Instant::now() + std::time::Duration::from_secs(60),
            crate::TurnCancellation::settlement_deadline,
        );
        match tokio::time::timeout_at(deadline, self.cleanup_turn_before(kind, deadline)).await {
            Ok(result) => result,
            Err(_) => Err(self
                .fenced_turn_failure(
                    anyhow::anyhow!("turn cleanup exceeded the shared settlement deadline"),
                    vec!["accepted writes remain owned; workspace and session recovery remain fenced".into()],
                )
                .into()),
        }
    }

    pub(crate) async fn cleanup_turn_before(
        &mut self,
        kind: crate::TurnCleanupKind,
        deadline: tokio::time::Instant,
    ) -> Result<crate::TurnCleanupResult> {
        self.disarm_btw_dispatcher();
        match kind {
            crate::TurnCleanupKind::Cancel { session } => {
                let killed = self
                    .quiesce_abnormal_turn_processes_before(deadline)
                    .await?;
                self.session_barrier().await?;
                match session {
                    crate::SessionRollback::AlreadyApplied => {
                        let _ = self.workspace.active_turn_message_start.take();
                    }
                    crate::SessionRollback::AgentOwned {
                        checkpoint_refs_before,
                    } => {
                        if let Err(error) =
                            self.rollback_turn_checkpoint(&checkpoint_refs_before).await
                        {
                            eprintln!(
                                "hi-agent: couldn't roll back cancelled workspace edits: {error:#}"
                            );
                        }
                        if let Some(start) = self.workspace.active_turn_message_start.take() {
                            self.truncate_messages(start);
                        }
                    }
                }
                let workspace_reconciled = self.reconcile_abnormal_turn_bounded().await;
                let (changes, current_workspace) = self.take_abnormal_turn_ledger_snapshot();
                self.settle_abnormal_workspace_operation(
                    hi_workspace::ExecutionDisposition::Cancelled,
                    workspace_reconciled,
                    &changes,
                    current_workspace,
                    false,
                )
                .await?;
                self.require_abnormal_turn_barrier(deadline).await?;
                self.persist_async().await?;
                self.session_barrier().await?;
                let outcome = self.finalize_cancelled_turn_with_changes(changes)?;
                Ok(crate::TurnCleanupResult {
                    outcome,
                    killed_backgrounds: killed,
                })
            }
            failure_kind @ (crate::TurnCleanupKind::Fail
            | crate::TurnCleanupKind::FailWithStopReason(_)) => {
                let stop_reason = match failure_kind {
                    crate::TurnCleanupKind::Fail => crate::TurnStopReason::InfrastructureFailure,
                    crate::TurnCleanupKind::FailWithStopReason(stop_reason) => stop_reason,
                    crate::TurnCleanupKind::Cancel { .. } => unreachable!(),
                };
                let killed = self
                    .quiesce_abnormal_turn_processes_before(deadline)
                    .await?;
                let workspace_reconciled = self.reconcile_abnormal_turn_bounded().await;
                let (changes, current_workspace) = self.take_abnormal_turn_ledger_snapshot();
                self.settle_abnormal_workspace_operation(
                    hi_workspace::ExecutionDisposition::Failed,
                    workspace_reconciled,
                    &changes,
                    current_workspace.clone(),
                    stop_reason.is_workspace_admission(),
                )
                .await?;
                if !stop_reason.is_workspace_admission() {
                    self.require_abnormal_turn_barrier(deadline).await?;
                }
                self.session_barrier().await?;
                let outcome = self.finalize_failed_turn_with_changes_and_reason(
                    changes,
                    workspace_reconciled,
                    current_workspace,
                    stop_reason,
                );
                Ok(crate::TurnCleanupResult {
                    outcome,
                    killed_backgrounds: killed,
                })
            }
        }
    }

    async fn require_abnormal_turn_barrier(&self, deadline: tokio::time::Instant) -> Result<()> {
        let mut preserved = self
            .runtime
            .background()
            .running_preserved_workspace_jobs(
                self.workspace
                    .active_turn_background_baseline
                    .as_deref()
                    .unwrap_or(&[]),
            )
            .await;
        if let Some(before) = &self.workspace.active_turn_task_baseline {
            for id in before {
                if let Some(job) = self.bg_tasks.candidate_workspace_job_id(id).await {
                    preserved.push(hi_workspace::JobId::new(job));
                }
            }
        }
        self.workspace_coordination
            .require_barrier_preserving_jobs(
                hi_workspace::BarrierKind::Publish,
                deadline.into_std(),
                &preserved,
            )
            .await?;
        Ok(())
    }

    /// Restore the checkpoint created by the active turn, if one is still on
    /// the stack, then put the exact pre-turn bounded undo history back.
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

        // If the interrupted tool still owns the admitted mutation, rollback
        // is part of that same operation and must not try to acquire a nested
        // permit. Its final restored image is committed by cleanup below.
        let restored_files = if self
            .workspace_coordination
            .active_parent_operation()
            .is_some()
        {
            self.undo_inner(false).await?.unwrap_or(0)
        } else {
            self.undo_without_ledger_reconcile().await?.unwrap_or(0)
        };
        if self.workspace.checkpoints == checkpoint_refs_before {
            return Ok(restored_files);
        }

        let checkpoints = checkpoint_refs_before.to_vec();
        self.write_session(move |session| session.record_checkpoints(&checkpoints))
            .await?;
        self.workspace.checkpoints = checkpoint_refs_before.to_vec();
        Ok(restored_files)
    }
}
