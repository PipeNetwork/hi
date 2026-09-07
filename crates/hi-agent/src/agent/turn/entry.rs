//! Turn entry, cancellation backstop, lifecycle callbacks, and project hooks.

use anyhow::{Context, Result};
use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, RunEvent,
    SemanticActivity,
};

use crate::{ReviewStatus, TurnOutcome, TurnStatus, TurnStopReason, Ui, VerificationStatus};

use super::helpers::effective_model_route;
use super::phase::TurnPhase;

/// Private control-flow marker used to leave the droppable turn body when it
/// observes whole-turn cancellation. Rollback must never run inside that body:
/// the outer cancellation backstop is allowed to drop it after a short grace,
/// whereas workspace restoration is not cancellation-safe once started.
#[derive(Debug)]
pub(in crate::agent::turn) struct TurnCancellationRequested;

impl std::fmt::Display for TurnCancellationRequested {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("turn cancellation requested")
    }
}

impl std::error::Error for TurnCancellationRequested {}

impl crate::Agent {
    /// Run one user turn to completion, emitting output through `ui`.
    ///
    /// Phases: [`TurnPhase::Setup`] → model/tool/steer loop →
    /// [`TurnPhase::WorkspaceRepair`] (optional stages; failures re-enter the
    /// model up to one initial check plus `max_verify_repairs` cycles) →
    /// [`TurnPhase::Settle`] → optional [`TurnPhase::Finalize`] →
    /// [`TurnPhase::Done`].
    pub async fn run_turn(&mut self, input: &str, ui: &mut dyn Ui) -> Result<TurnOutcome> {
        self.run_turn_cancellable(input, ui, crate::TurnCancellation::new())
            .await
    }

    /// Run one user turn with a frontend-owned cancellation signal.
    ///
    /// The configured hard turn timeout is still enforced on this path; GUI
    /// and CLI frontends must not have to choose between cooperative Ctrl-C
    /// cleanup and the configured deadline.
    pub async fn run_turn_cancellable(
        &mut self,
        input: &str,
        ui: &mut dyn Ui,
        cancellation: crate::TurnCancellation,
    ) -> Result<TurnOutcome> {
        let configured_timeout = self.config.loop_limits.turn_timeout;
        let mut turn = Box::pin(self.run_turn_cancellable_inner(input, ui, cancellation.clone()));
        if let Some(timeout) = configured_timeout {
            match tokio::time::timeout(timeout, turn.as_mut()).await {
                Ok(result) => result,
                Err(_) => {
                    cancellation.cancel();
                    // The Agent's inner owner enforces the same deadline and
                    // publishes its receipt. Keep polling it through cleanup.
                    let result = turn.as_mut().await;
                    match result {
                        Ok(outcome) if outcome.stop_reason == TurnStopReason::Cancelled => {
                            Err(crate::TurnFailure::new(
                                anyhow::anyhow!(
                                    "turn deadline exceeded after {}s",
                                    timeout.as_secs()
                                ),
                                outcome,
                            )
                            .into())
                        }
                        result => result,
                    }
                }
            }
        } else {
            turn.as_mut().await
        }
    }

    async fn run_turn_cancellable_inner(
        &mut self,
        input: &str,
        ui: &mut dyn Ui,
        cancellation: crate::TurnCancellation,
    ) -> Result<TurnOutcome> {
        let requested_drive_kind = crate::DriveKind::from_prompt(input);
        if let Err(error) = self.ensure_session_reusable() {
            let mut outcome = self.finalize_failed_turn_snapshot_only();
            outcome.stop_reason = TurnStopReason::InfrastructureFailure;
            let mut failure = crate::TurnFailure::new(error, outcome);
            failure.settlement_pending = true;
            return Err(failure.into());
        }
        let task_baseline = self.bg_tasks.list().await;
        self.workspace.active_turn_task_baseline = Some(task_baseline.clone());
        // Reports are telemetry. Returned receipts own terminal semantics.
        self.report.last_turn_outcome = None;
        self.report.terminal_input_digest = None;
        self.report.provisional_goal_baseline = None;
        // A permit admitted before this call belongs to its existing owner.
        // Reject before starting a body, so failure cleanup cannot consume it.
        if self
            .workspace_coordination
            .active_parent_operation()
            .is_some()
        {
            self.workspace.active_turn_task_baseline = None;
            let route = self.report.last_effective_route.clone();
            let outcome = TurnOutcome::workspace_admission_blocked(
                route.model,
                route.provider,
                Vec::new(),
                TurnStopReason::WorkspaceNotReady,
            );
            self.report.set_outcome(outcome.clone());
            return Err(crate::TurnFailure::new(
                anyhow::anyhow!("a workspace mutation is already admitted and awaiting settlement"),
                outcome,
            )
            .into());
        }
        let foreground_processes = self.foreground_process_registry();
        let background_processes = self.runtime.background_arc();
        let background_baseline = background_processes.ids();
        self.workspace.active_turn_background_baseline = Some(background_baseline.clone());
        let background_tasks = self.bg_tasks.clone();
        let signal_processes = || async {
            foreground_processes.kill_current();
            background_processes.kill_started_after(&background_baseline);
            background_tasks.signal_started_after(&task_baseline).await;
        };
        if cancellation.is_cancelled() {
            signal_processes().await;
            let deadline = cancellation.settlement_deadline();
            let cleanup = async {
                let restore_plan_pause = requested_drive_kind == crate::DriveKind::Plan
                    || self.pending_plan_interruption_resume
                    || self.turn_consumed_plan_interruption;
                let pause_result = if restore_plan_pause {
                    self.pause_plan_drive_until_user_input_async()
                        .await
                        .map(|_| ())
                        .context("persisting plan interruption before cancellation cleanup")
                } else {
                    Ok(())
                };
                self.pending_plan_interruption_resume = false;
                self.turn_consumed_plan_interruption = false;
                let deadline = cancellation.settlement_deadline();
                let cleanup = self.cleanup_turn_before(
                    crate::TurnCleanupKind::Cancel {
                        session: crate::SessionRollback::AgentOwned {
                            checkpoint_refs_before: self.checkpoint_refs().to_vec(),
                        },
                    },
                    deadline,
                );
                let outcome = cleanup.await?.outcome;
                pause_result?;
                Ok(outcome)
            };
            let result: Result<TurnOutcome> = tokio::time::timeout_at(deadline, cleanup)
                .await
                .map_err(|_| anyhow::anyhow!("turn cancellation settlement deadline exceeded"))
                .and_then(|result| result);
            let result = match result {
                Ok(outcome) => Ok(outcome),
                Err(error) => Err(self.fenced_turn_failure(error, Vec::new()).into()),
            };
            return self.publish_terminal_receipt(result, deadline, None).await;
        }
        let message_count_before = self.messages.len();
        let state_before = self.state_snapshot();
        let checkpoint_refs_before = self.checkpoint_refs().to_vec();
        // Install the turn cancel flag before the body runs so tool batches and
        // the Model→Tools loop can cooperatively abort, synthesize tool_results,
        // and return a private marker to the outer cleanup owner instead of only
        // dying on drop.
        self.turn_cancellation = Some(cancellation.clone());
        // Soft deadline after cancel is observed: keep polling the body so it
        // can settle; then drop it and force cleanup (hard backstop). The body
        // future must be dropped before cleanup_turn borrows `self` again.
        const COOPERATIVE_CANCEL_GRACE: std::time::Duration = std::time::Duration::from_millis(750);
        let interrupt = std::sync::Arc::clone(&self.interrupt);
        let (body_result, cancellation_observed, foreground_reap_proven, settlement_expired) = {
            let body = self.run_turn_body(input, ui);
            tokio::pin!(body);
            tokio::select! {
                biased;
                result = &mut body => (Some(result), false, true, false),
                _ = cancellation.wait_for_cancellation() => {
                    interrupt.store(true, std::sync::atomic::Ordering::Release);
                    // Wake an in-flight foreground shell/program immediately.
                    // Its owning capture future remains alive during the grace
                    // window so it can observe exit and reap the direct child.
                    signal_processes().await;
                    let mut result = tokio::select! {
                        biased;
                        result = &mut body => Some(result),
                        _ = tokio::time::sleep_until(std::cmp::min(
                            tokio::time::Instant::now() + COOPERATIVE_CANCEL_GRACE,
                            cancellation.settlement_deadline(),
                        )) => None,
                    };
                    let mut foreground_reap_proven = true;
                    // Catch jobs admitted while the cooperative body was still
                    // polling, before any longer native-reap or storage wait.
                    signal_processes().await;
                    if result.is_none() && foreground_processes.active_count() > 0 {
                        // Keep the body allocation alive while capture owns the
                        // child. If we dropped it first, the registry token
                        // would disappear before Tokio's kill-on-drop reaper
                        // could be observed, creating a false quiescent state.
                        const FOREGROUND_REAP_GRACE: std::time::Duration =
                            std::time::Duration::from_secs(5);
                        tokio::select! {
                            biased;
                            settled = &mut body => result = Some(settled),
                            reaped = foreground_processes.wait_until_empty(std::cmp::min(
                                FOREGROUND_REAP_GRACE,
                                cancellation.settlement_deadline().saturating_duration_since(tokio::time::Instant::now()),
                            )) => {
                                foreground_reap_proven = reaped;
                            }
                        }
                    }
                    // Cancellation has won the outer race. Even if the body
                    // happens to finish normally during its grace window, the
                    // caller's request still owns the result and must roll the
                    // turn back. A normal result that wins the biased outer
                    // branch is not retroactively cancelled.
                    (result, true, foreground_reap_proven, false)
                }
                _ = async {
                    let deadline = cancellation.wait_for_settlement_deadline().await;
                    tokio::time::sleep_until(deadline).await;
                } => {
                    cancellation.cancel();
                    interrupt.store(true, std::sync::atomic::Ordering::Release);
                    signal_processes().await;
                    (None, false, foreground_processes.active_count() == 0, true)
                }
            }
            // `body` drops here, releasing `&mut self`.
        };
        self.turn_cancellation = None;
        self.interrupt
            .store(false, std::sync::atomic::Ordering::Release);
        self.finish_drive_turn();
        let forced_abort = body_result.is_none();
        let cooperative_cancel = body_result.as_ref().is_some_and(|result| {
            result
                .as_ref()
                .err()
                .is_some_and(|error| error.is::<TurnCancellationRequested>())
        });
        let cancellation_cleanup = cancellation_observed || forced_abort || cooperative_cancel;
        if cancellation_cleanup {
            // Plan pause persistence can queue behind an accepted slow append.
            // Signal all turn-owned executions before awaiting that commit.
            signal_processes().await;
        }
        let cancellation_abort_reason = cancellation_cleanup.then(|| {
            cancellation
                .abort_reason()
                .unwrap_or(hi_agent_lifecycle::TurnAbortReason::Interrupted)
        });
        let result: Result<TurnOutcome> = if settlement_expired {
            Err(self.fenced_turn_failure(
                anyhow::anyhow!("turn settlement deadline exceeded after 60s"),
                vec!["accepted publication remains owned; unresolved workspace effects remain fenced".into()],
            ).into())
        } else if cancellation_cleanup {
            let deadline = cancellation.settlement_deadline();
            let cleanup = async {
                // Persist the stop latch before rewriting the transcript. If the
                // process dies between these appends, restart remains safely
                // paused instead of autonomously re-running abandoned plan work.
                let restore_plan_pause = requested_drive_kind == crate::DriveKind::Plan
                    || self.pending_plan_interruption_resume
                    || self.turn_consumed_plan_interruption;
                let pause_result = if restore_plan_pause {
                    self.pause_plan_drive_until_user_input_async()
                        .await
                        .map(|_| ())
                        .context("persisting plan interruption before cancelled-turn rewind")
                } else {
                    Ok(())
                };
                // This is the sole owner of cancellation rollback. In particular,
                // it is outside the body future that the cooperative grace may
                // drop, so an in-progress checkpoint restore can never be detached
                // and then re-entered by a second cleanup attempt.
                // A killed child can execute a final write until it is reaped.
                // Never begin rollback while foreground or auto-backgrounded turn
                // writers can still race the restoration.
                let quiescence_result = self.quiesce_abnormal_turn_processes_before(deadline).await;
                let quiescence_error = if foreground_reap_proven {
                    quiescence_result.err()
                } else {
                    Some(anyhow::anyhow!(
                        "timed out waiting for a cancelled foreground process to be reaped"
                    ))
                };
                if quiescence_error.is_some() {
                    // The permit's synchronous drop fence records the ambiguity
                    // even if the remaining cleanup is itself cancelled later.
                    let _ = self.workspace_coordination.abandon_active();
                }
                // Drain accepted session commands (including their metadata
                // exports) before rolling workspace bytes back.
                self.session_barrier()
                    .await
                    .context("draining session writes before rollback")?;
                let workspace_rolled_back = if quiescence_error.is_none() {
                    self.rollback_turn_checkpoint(&checkpoint_refs_before)
                        .await
                        .context("rolling back cancelled workspace edits")?
                        > 0
                } else {
                    false
                };
                let message_start = self
                    .workspace
                    .active_turn_message_start
                    .unwrap_or(message_count_before);
                if let Err(error) = self
                    .rewind_to_snapshot_durable_with_workspace_rollback_async(
                        message_start,
                        &state_before,
                        workspace_rolled_back,
                    )
                    .await
                {
                    // Keep the live agent coherent even when its durable sink is
                    // unavailable. This mirrors the interactive interrupt path;
                    // cleanup below still finalizes cancellation and surfaces a
                    // persistence error if its final write also fails.
                    eprintln!("hi-agent: couldn't persist cancelled turn discard: {error:#}");
                    self.truncate_messages(message_start);
                    self.restore_state_snapshot_with_workspace_rollback(
                        &state_before,
                        workspace_rolled_back,
                    );
                    return Err(error.context("persisting cancelled turn discard"));
                }
                if let Some(error) = quiescence_error {
                    pause_result?;
                    Err(error
                        .context("cancelled turn could not prove all writer processes were reaped"))
                } else {
                    let cleanup_result = self
                        .cleanup_turn_before(
                            crate::TurnCleanupKind::Cancel {
                                session: crate::SessionRollback::AlreadyApplied,
                            },
                            deadline,
                        )
                        .await
                        .map(|cleanup| cleanup.outcome);
                    cleanup_result.and_then(|outcome| {
                        pause_result?;
                        Ok(outcome)
                    })
                }
            };
            match tokio::time::timeout_at(deadline, cleanup).await {
                Ok(Ok(outcome)) => Ok(outcome),
                Ok(Err(error)) => Err(self.fenced_turn_failure(error, Vec::new()).into()),
                Err(_) => Err(self
                    .fenced_turn_failure(
                        anyhow::anyhow!("turn cancellation settlement deadline exceeded"),
                        vec![
                            "accepted settlement continues; workspace recovery remains fenced"
                                .into(),
                        ],
                    )
                    .into()),
            }
        } else {
            match body_result.expect("non-cancelled turn body must have a result") {
                Ok(outcome) => Ok(outcome),
                Err(original)
                    if crate::TurnFailure::from_error(&original)
                        .is_some_and(crate::TurnFailure::body_settled) =>
                {
                    Err(original)
                }
                Err(original) => {
                    let deadline = cancellation.settlement_deadline();
                    let kind = crate::TurnCleanupKind::for_error(&original);
                    match tokio::time::timeout_at(deadline, self.cleanup_turn_before(kind, deadline)).await {
                        Ok(Ok(cleanup)) => Err(crate::TurnFailure::new(original, cleanup.outcome).into()),
                        Ok(Err(cleanup)) => Err(self.fenced_turn_failure(original, vec![format!("{cleanup:#}")]).into()),
                        Err(_) => Err(self.fenced_turn_failure(original, vec!["turn settlement deadline exceeded; accepted settlement remains owned".into()]).into()),
                    }
                }
            }
        };
        let drive_must_pause = match &result {
            Err(_) => true,
            Ok(outcome) => crate::plan_drive::outcome_blocks_automatic_drive(outcome),
        };
        let drive_state_result =
            tokio::time::timeout_at(cancellation.settlement_deadline(), async {
                let mut drive_state_result = self
                    .settle_plan_interruption_resume_async(!drive_must_pause)
                    .await
                    .context("settling transactional plan interruption resume");
                if drive_state_result.is_ok()
                    && drive_must_pause
                    && requested_drive_kind == crate::DriveKind::Plan
                {
                    drive_state_result = self
                        .pause_plan_drive_until_user_input_async()
                        .await
                        .map(|_| ())
                        .context("pausing plan drive after unsuccessful synthetic turn");
                }
                drive_state_result
            })
            .await
            .map_err(|_| {
                self.session_recovery_pending |= self.has_session_io();
                anyhow::anyhow!("drive-state persistence exceeded the shared settlement deadline")
            })
            .and_then(|result| result);
        let result = match drive_state_result {
            Ok(()) => result,
            Err(error) => match result {
                Ok(outcome) => Err(crate::TurnFailure::new(error, outcome).into()),
                Err(original) => match original.downcast::<crate::TurnFailure>() {
                    Ok(mut failure) => {
                        failure.cleanup_diagnostics.push(format!("{error:#}"));
                        Err(failure.into())
                    }
                    Err(original) => Err(self
                        .fenced_turn_failure(original, vec![format!("{error:#}")])
                        .into()),
                },
            },
        };
        let abort_reason = cancellation_abort_reason.or_else(|| match &result {
            Ok(outcome) if outcome.status == TurnStatus::Cancelled => cancellation
                .abort_reason()
                .or(Some(hi_agent_lifecycle::TurnAbortReason::Interrupted)),
            _ => None,
        });
        // Terminal bookkeeping lives outside `run_turn_body`: the body is the
        // future deliberately dropped by the hard cancellation backstop, so it
        // cannot own lifecycle callbacks, Done phase, turn count, or terminal
        // semantic events without skipping them on an uncooperative provider.
        let deadline = cancellation.settlement_deadline();
        let terminal_workspace_baseline = self
            .report
            .terminal_input_digest
            .take()
            .unwrap_or_else(|| self.runtime.ledger().workspace_revision());
        let finalized = tokio::time::timeout_at(
            deadline,
            self.finalize_turn_result(
                input,
                ui,
                &result,
                abort_reason,
                forced_abort,
                &cancellation,
            ),
        )
        .await;
        let mut result = if finalized.is_err() {
            let detail = "terminal turn notices exceeded the shared settlement deadline";
            match result {
                Ok(outcome) => {
                    let mut failure = crate::TurnFailure::new(anyhow::anyhow!(detail), outcome);
                    let _ = self.workspace_coordination.abandon_active();
                    let status = self.workspace_controller_status();
                    failure.settlement_pending = !matches!(
                        status.state,
                        hi_workspace::WorkspaceState::Ready
                            | hi_workspace::WorkspaceState::LocalAuditDegraded
                    ) || !status.active_jobs.is_empty()
                        || self.foreground_process_registry().active_count() != 0;
                    self.foreground_process_registry().kill_current();
                    Err(failure.into())
                }
                Err(error) => match error.downcast::<crate::TurnFailure>() {
                    Ok(mut failure) => {
                        failure.cleanup_diagnostics.push(detail.into());
                        Err(failure.into())
                    }
                    Err(error) => Err(self.fenced_turn_failure(error, vec![detail.into()]).into()),
                },
            }
        } else {
            result
        };

        let final_evidence = tokio::time::timeout_at(
            deadline,
            self.reconcile_terminal_workspace(
                &mut result,
                &state_before,
                &terminal_workspace_baseline,
                ui,
            ),
        )
        .await;
        let evidence_pending = final_evidence.is_err()
            || (matches!(&final_evidence, Ok(Err(_)))
                && !matches!(
                    self.workspace_controller_status().state,
                    hi_workspace::WorkspaceState::Ready
                        | hi_workspace::WorkspaceState::LocalAuditDegraded
                ));
        let recovery_credit = match final_evidence {
            Ok(Ok(credit)) => credit,
            failed => {
                let error = match failed {
                    Ok(Err(error)) => error,
                    Err(_) => {
                        self.session_recovery_pending |= self.has_session_io();
                        let _ = self.workspace_coordination.abandon_active();
                        anyhow::anyhow!(
                            "final workspace evidence exceeded the shared settlement deadline"
                        )
                    }
                    Ok(Ok(_)) => unreachable!(),
                };
                result = match result {
                    Ok(outcome) => {
                        let mut failure = crate::TurnFailure::new(error, outcome);
                        failure.invalidate_terminal_verification();
                        failure.settlement_pending |= evidence_pending;
                        Err(failure.into())
                    }
                    Err(original) => {
                        let mut failure = original
                            .downcast::<crate::TurnFailure>()
                            .expect("entry owns a typed failure receipt");
                        failure.invalidate_terminal_verification();
                        failure.cleanup_diagnostics.push(format!("{error:#}"));
                        failure.settlement_pending |= evidence_pending;
                        Err(failure.into())
                    }
                };
                None
            }
        };
        if result.is_err() {
            self.emit_deterministic_closeout(ui);
        }
        let event_turn = if matches!(&result, Ok(outcome) if outcome.stop_reason == TurnStopReason::TurnLimit)
        {
            self.turn_count.saturating_add(1)
        } else {
            self.turn_count
        };
        let result = self
            .publish_terminal_receipt(result, deadline, recovery_credit)
            .await;
        self.report.provisional_goal_baseline = None;
        self.emit_terminal_result_event(ui, &result, event_turn);
        result
    }

    /// Timeout paths never claim successful cleanup. Admitted permits publish
    /// recovery on drop; shielded settlements retain their existing owner.
    pub(crate) fn fenced_turn_failure(
        &mut self,
        original: anyhow::Error,
        cleanup_diagnostics: Vec<String>,
    ) -> crate::TurnFailure {
        self.session_recovery_pending |= self.has_session_io();
        let tasks = self.bg_tasks.clone();
        let before = self
            .workspace
            .active_turn_task_baseline
            .clone()
            .unwrap_or_default();
        // Retain a cancellation owner even when the caller's budget is already
        // exhausted. It signals all executions and leaves accepted callbacks
        // with their existing monitor rather than dropping their publication.
        tokio::spawn(async move {
            tasks
                .kill_started_after_before(&before, tokio::time::Instant::now())
                .await;
        });
        let _ = self.workspace_coordination.abandon_active();
        self.foreground_process_registry().kill_current();
        let outcome = self.finalize_failed_turn_snapshot_only();
        let mut failure = crate::TurnFailure::new(original, outcome);
        failure.cleanup_diagnostics = cleanup_diagnostics;
        failure.settlement_pending = true;
        failure
    }

    async fn run_turn_body(&mut self, input: &str, ui: &mut dyn Ui) -> Result<TurnOutcome> {
        // A prior timed-out waiter can leave an accepted append with the owner.
        // Do not admit a new turn until its durable result has been observed.
        self.session_barrier().await?;
        ui.semantic_event(RunEvent::new(
            EventKind::RunStarted,
            EventContext::default(),
            SemanticActivity {
                verb: ActivityVerb::Start,
                object: ActivityObject::Run,
                state: ActivityState::Running,
                group_key: format!("run:turn:{}", self.turn_count.saturating_add(1)),
                title: "Run started".into(),
                detail: None,
                refs: Vec::new(),
                progress: None,
            },
        ));
        // Per-session turn limit (`/turns <n>`). Checked after the semantic
        // RunStarted event, but before lifecycle extensions, hooks, or model/tool
        // work start. `None` = unlimited (the default).
        if let Some(limit) = self.config.max_turns
            && self.turn_count >= limit
        {
            // Per-session turn limit reached before this turn started.
            let outcome = TurnOutcome {
                status: TurnStatus::Completed,
                verification: VerificationStatus::NotApplicable,
                review: ReviewStatus::NotRequired,
                stop_reason: TurnStopReason::TurnLimit,
                changed_files: Vec::new(),
                verified_workspace_revision: None,
                effective_route: effective_model_route(
                    &self.config,
                    Some(self.report.last_effective_route.model.as_str()),
                ),
                review_same_model: self.skeptic_shares_session_model(),
                leftover: None,
                plan_leftover: None,
            };
            self.report.set_outcome(outcome.clone());
            return Ok(outcome);
        }
        // Pair every started body with exactly one terminal callback from
        // `finalize_turn_result`, including preflight errors and hard-backstop
        // cancellation that drops this future. A turn-limit rejection above is
        // not a started turn and deliberately dispatches no lifecycle callback.
        if let Some(registry) = &self.extensions {
            for contributor in registry.turn_lifecycle_contributors() {
                contributor
                    .on_turn_start(&hi_agent_lifecycle::TurnStartInput::new(false))
                    .await;
            }
        }
        if self.config.execution.is_durable() && self.session.is_none() {
            anyhow::bail!(
                "durable execution requires a persisted session; remove --no-save or install a SessionSink"
            );
        }
        // User lifecycle hooks are intentionally outside the model/tool loop.
        // `pre-turn` is a gate; `post-turn` and `stop` are best-effort notices.
        let hooks = self.workspace_root().join(".hi/hooks");
        let hooks_trusted = crate::workspace_trusted(self.workspace_root());
        let portable_workspace = self.pipefs_workspace_active();
        if hooks.join("pre-turn").is_file() && hooks_trusted && !portable_workspace {
            let hook_cancellation = self
                .turn_cancellation
                .clone()
                .expect("turn cancellation is installed before lifecycle hooks");
            match self
                .run_workspace_lifecycle_hook("pre-turn", input, &hook_cancellation)
                .await
            {
                Ok(Some(report)) => ui.status(&report),
                Ok(None) => return Err(TurnCancellationRequested.into()),
                Err(error) => {
                    return Err(anyhow::anyhow!("pre-turn hook blocked turn: {error:#}"));
                }
            }
        } else if hooks.join("pre-turn").is_file() && portable_workspace {
            ui.status(
                "project hooks skipped in PipeFS; trust and execute restored code explicitly",
            );
        } else if hooks.join("pre-turn").is_file() {
            ui.status("project hooks skipped: workspace untrusted (run /trust on to enable)");
        }
        self.run_turn_core(input, ui).await
    }

    async fn finalize_turn_result(
        &mut self,
        input: &str,
        ui: &mut dyn Ui,
        result: &Result<TurnOutcome>,
        abort_reason: Option<hi_agent_lifecycle::TurnAbortReason>,
        forced_abort: bool,
        cancellation: &crate::TurnCancellation,
    ) {
        let turn_limit = matches!(
            result,
            Ok(outcome) if outcome.stop_reason == TurnStopReason::TurnLimit
        );
        // Stamp terminal state before any best-effort callback. In particular,
        // a slow extension cannot leave a hard-cancelled turn visibly stuck in
        // Model/Tools after the cancellation cleanup already completed.
        if !turn_limit {
            self.turn_count = self.turn_count.saturating_add(1);
        }
        self.set_turn_phase(TurnPhase::Done);
        // Exactly one terminal callback per body start. Cancellation wins over
        // a cleanup error because it, not model failure, ended that turn.
        if !turn_limit && let Some(registry) = &self.extensions {
            // Start every best-effort terminal notice exactly once,
            // concurrently, under one small global budget. Done/error hooks
            // need the same bound as abort hooks: a turn timeout can fire after
            // the body settles, and a wedged extension must not strand it.
            const TERMINAL_LIFECYCLE_GRACE: std::time::Duration =
                std::time::Duration::from_millis(100);
            let contributors = registry.turn_lifecycle_contributors().to_vec();
            let callbacks = async {
                if let Some(reason) = abort_reason {
                    let input = hi_agent_lifecycle::TurnAbortInput { reason };
                    futures_util::future::join_all(
                        contributors
                            .iter()
                            .map(|contributor| contributor.on_turn_abort(&input)),
                    )
                    .await;
                } else if let Err(error) = result {
                    let message = format!("{error:#}");
                    let input = hi_agent_lifecycle::TurnErrorInput { message: &message };
                    futures_util::future::join_all(
                        contributors
                            .iter()
                            .map(|contributor| contributor.on_turn_error(&input)),
                    )
                    .await;
                } else {
                    let input = hi_agent_lifecycle::TurnDoneInput;
                    futures_util::future::join_all(
                        contributors
                            .iter()
                            .map(|contributor| contributor.on_turn_done(&input)),
                    )
                    .await;
                }
            };
            let _ = tokio::time::timeout(TERMINAL_LIFECYCLE_GRACE, callbacks).await;
        }

        // Hooks have no implicit productive timeout. The hard 750ms turn
        // cancellation backstop drops the hook future, whose process-group
        // guard owns descendant cleanup, so do not start more hooks after the
        // live body was already force-dropped.
        if forced_abort || cancellation.is_cancelled() {
            return;
        }

        let summary = match result {
            Ok(outcome) => format!("status=ok\noutcome={outcome:?}\ninput={input}"),
            Err(error) => format!("status=error\nerror={error:#}\ninput={input}"),
        };
        let hooks = self.workspace_root().join(".hi/hooks");
        let hooks_trusted = crate::workspace_trusted(self.workspace_root());
        let portable_workspace = self.pipefs_workspace_active();
        if hooks.join("post-turn").is_file() && hooks_trusted && !portable_workspace {
            match self
                .run_workspace_lifecycle_hook("post-turn", &summary, cancellation)
                .await
            {
                Ok(Some(report)) => ui.status(&report),
                Err(error) => ui.status(&format!("post-turn hook failed: {error:#}")),
                Ok(None) => return,
            }
        }
        if hooks.join("stop").is_file() && hooks_trusted && !portable_workspace {
            match self
                .run_workspace_lifecycle_hook("stop", &summary, cancellation)
                .await
            {
                Ok(Some(report)) => ui.status(&report),
                Err(error) => ui.status(&format!("stop hook failed: {error:#}")),
                Ok(None) => (),
            }
        }
    }
}
