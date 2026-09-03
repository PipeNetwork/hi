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
        if let Some(timeout) = self.config.loop_limits.turn_timeout {
            // Keep the very large turn state machine behind one pointer. In
            // debug builds an inline copy here is also embedded in every
            // caller's future (including multi-case async tests), which can
            // exhaust the default test-thread stack before the first poll.
            // Boxing also guarantees the timeout and cancellation-settlement
            // branches keep polling the exact same future allocation.
            let mut turn =
                Box::pin(self.run_turn_cancellable_inner(input, ui, cancellation.clone()));
            match tokio::time::timeout(timeout, turn.as_mut()).await {
                Ok(result) => result,
                Err(_) => {
                    // Do not drop a live turn future: that bypasses its cleanup
                    // path and can leave cancellation flags, transcript/ledger
                    // baselines, and turn-scoped background jobs attached to the
                    // reusable Agent. Signal the same future and let its bounded
                    // cooperative-cancel path settle before reporting timeout.
                    cancellation.cancel();
                    const DEADLINE_SETTLEMENT_GRACE: std::time::Duration =
                        std::time::Duration::from_secs(30);
                    let settled =
                        tokio::time::timeout(DEADLINE_SETTLEMENT_GRACE, turn.as_mut()).await;
                    let settled = match settled {
                        Ok(settled) => settled,
                        Err(_) => {
                            // A rollback/cleanup implementation must not turn the
                            // hard deadline into another unbounded wait. Drop the
                            // in-flight cleanup future and leave an explicit
                            // terminal diagnostic; the workspace state is
                            // intentionally reported as uncertain.
                            drop(turn);
                            self.turn_cancellation = None;
                            self.interrupt
                                .store(false, std::sync::atomic::Ordering::Release);
                            self.finish_drive_turn();
                            let _ = self.kill_turn_backgrounds();
                            if crate::DriveKind::from_prompt(input) == crate::DriveKind::Plan
                                || self.pending_plan_interruption_resume
                                || self.turn_consumed_plan_interruption
                            {
                                self.pause_plan_drive_until_user_input().context(
                                    "persisting plan interruption after cleanup deadline",
                                )?;
                            }
                            ui.assistant_text(
                                "The turn hit its hard deadline and cleanup did not settle in time. I stopped waiting; the workspace may contain partial changes and should be inspected before continuing.",
                            );
                            ui.assistant_end();
                            anyhow::bail!(
                                "turn deadline exceeded after {}s; cleanup exceeded its {}s grace period",
                                timeout.as_secs(),
                                DEADLINE_SETTLEMENT_GRACE.as_secs()
                            );
                        }
                    };
                    match settled {
                        Ok(outcome) if outcome.stop_reason == TurnStopReason::Cancelled => {
                            anyhow::bail!("turn deadline exceeded after {}s", timeout.as_secs())
                        }
                        // The body can win and commit immediately before the
                        // deadline while a bounded lifecycle callback or
                        // post-turn hook is still settling. In that case the
                        // token only skips/cancels best-effort notices; report
                        // the already-committed body result instead of claiming
                        // that its transcript/workspace were rolled back.
                        settled => settled,
                    }
                }
            }
        } else {
            Box::pin(self.run_turn_cancellable_inner(input, ui, cancellation)).await
        }
    }

    async fn run_turn_cancellable_inner(
        &mut self,
        input: &str,
        ui: &mut dyn Ui,
        cancellation: crate::TurnCancellation,
    ) -> Result<TurnOutcome> {
        let requested_drive_kind = crate::DriveKind::from_prompt(input);
        // Pin the currently active decision-engine generation for the whole
        // turn. A reload requested during streaming becomes pending and is
        // promoted only after this lease is dropped.
        let engine_lease = match self
            .engine_runtime
            .begin_turn()
            .context("pinning decision engine generation")
        {
            Ok(lease) => lease,
            Err(error) => {
                if requested_drive_kind == crate::DriveKind::Plan
                    || self.pending_plan_interruption_resume
                    || self.turn_consumed_plan_interruption
                {
                    self.pause_plan_drive_until_user_input().context(
                        "restoring plan interruption after decision-engine setup failed",
                    )?;
                }
                return Err(error);
            }
        };
        // A frontend may use the terminal report to decide whether an Err
        // already passed through Agent-owned cancellation cleanup (notably the
        // hard-timeout path, which preserves its deadline error). Clear the
        // previous turn before any await/preflight so a late Ctrl-C cannot make
        // a new early error look like an already-finalized cancellation.
        self.report.last_turn_outcome = None;
        if cancellation.is_cancelled() {
            let restore_plan_pause = requested_drive_kind == crate::DriveKind::Plan
                || self.pending_plan_interruption_resume
                || self.turn_consumed_plan_interruption;
            let pause_result = if restore_plan_pause {
                self.pause_plan_drive_until_user_input()
                    .map(|_| ())
                    .context("persisting plan interruption before cancellation cleanup")
            } else {
                Ok(())
            };
            self.pending_plan_interruption_resume = false;
            self.turn_consumed_plan_interruption = false;
            let cleanup_result = self
                .cleanup_turn(crate::TurnCleanupKind::Cancel {
                    session: crate::SessionRollback::AgentOwned {
                        checkpoint_refs_before: self.checkpoint_refs().to_vec(),
                    },
                })
                .await
                .map(|cleanup| cleanup.outcome);
            return cleanup_result.and_then(|outcome| {
                pause_result?;
                Ok(outcome)
            });
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
        let (body_result, cancellation_observed) = {
            let body = self.run_turn_body(input, ui, engine_lease);
            tokio::pin!(body);
            tokio::select! {
                biased;
                result = &mut body => (Some(result), false),
                _ = wait_for_turn_cancellation(cancellation.clone()) => {
                    interrupt.store(true, std::sync::atomic::Ordering::Release);
                    let result = tokio::select! {
                        biased;
                        result = &mut body => Some(result),
                        _ = tokio::time::sleep(COOPERATIVE_CANCEL_GRACE) => None,
                    };
                    // Cancellation has won the outer race. Even if the body
                    // happens to finish normally during its grace window, the
                    // caller's request still owns the result and must roll the
                    // turn back. A normal result that wins the biased outer
                    // branch is not retroactively cancelled.
                    (result, true)
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
        let cancellation_abort_reason = cancellation_cleanup.then(|| {
            cancellation
                .abort_reason()
                .unwrap_or(hi_agent_lifecycle::TurnAbortReason::Interrupted)
        });
        let result = if cancellation_cleanup {
            // Persist the stop latch before rewriting the transcript. If the
            // process dies between these appends, restart remains safely
            // paused instead of autonomously re-running abandoned plan work.
            let restore_plan_pause = requested_drive_kind == crate::DriveKind::Plan
                || self.pending_plan_interruption_resume
                || self.turn_consumed_plan_interruption;
            let pause_result = if restore_plan_pause {
                self.pause_plan_drive_until_user_input()
                    .map(|_| ())
                    .context("persisting plan interruption before cancelled-turn rewind")
            } else {
                Ok(())
            };
            // This is the sole owner of cancellation rollback. In particular,
            // it is outside the body future that the cooperative grace may
            // drop, so an in-progress checkpoint restore can never be detached
            // and then re-entered by a second cleanup attempt.
            let workspace_rolled_back = match self
                .rollback_turn_checkpoint(&checkpoint_refs_before)
                .await
            {
                Ok(restored_files) => restored_files > 0,
                Err(error) => {
                    eprintln!("hi-agent: couldn't roll back cancelled workspace edits: {error:#}");
                    false
                }
            };
            let message_start = self
                .workspace
                .active_turn_message_start
                .unwrap_or(message_count_before);
            if let Err(error) = self.rewind_to_snapshot_durable_with_workspace_rollback(
                message_start,
                &state_before,
                workspace_rolled_back,
            ) {
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
            }
            let cleanup_result = self
                .cleanup_turn(crate::TurnCleanupKind::Cancel {
                    session: crate::SessionRollback::AlreadyApplied,
                })
                .await
                .map(|cleanup| cleanup.outcome);
            cleanup_result.and_then(|outcome| {
                pause_result?;
                Ok(outcome)
            })
        } else {
            body_result.expect("non-cancelled turn body must have a result")
        };
        let drive_must_pause = match &result {
            Err(_) => true,
            Ok(outcome) => crate::plan_drive::outcome_blocks_automatic_drive(outcome),
        };
        let mut drive_state_result = self
            .settle_plan_interruption_resume(!drive_must_pause)
            .context("settling transactional plan interruption resume");
        if drive_state_result.is_ok()
            && drive_must_pause
            && requested_drive_kind == crate::DriveKind::Plan
        {
            drive_state_result = self
                .pause_plan_drive_until_user_input()
                .map(|_| ())
                .context("pausing plan drive after unsuccessful synthetic turn");
        }
        let result = match drive_state_result {
            Ok(()) => result,
            Err(error) => Err(error),
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
        self.finalize_turn_result(
            input,
            ui,
            &result,
            abort_reason,
            forced_abort,
            &cancellation,
        )
        .await;

        match result {
            Ok(outcome) => Ok(outcome),
            Err(error) => {
                // Kill turn-scoped backgrounds before surfacing the error — a
                // mid-turn provider/tool failure must not leak delegate/explore
                // subagents started this turn. Only the background kill runs
                // here; ledger reconciliation and `last_changed_files` stay
                // with the caller's own `cleanup_turn(Fail)` /
                // `finalize_failed_turn` (idempotent via `.take()` on the
                // baseline), preserving the contract frontends rely on.
                let _ = self.kill_turn_backgrounds();
                Err(error)
            }
        }
    }

    async fn run_turn_body(
        &mut self,
        input: &str,
        ui: &mut dyn Ui,
        engine_lease: hi_engine_host::EngineLease,
    ) -> Result<TurnOutcome> {
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
        if hooks.join("pre-turn").is_file() && hooks_trusted {
            let report = crate::run_hook(self.workspace_root(), "pre-turn", input)
                .await
                .map_err(|e| anyhow::anyhow!("pre-turn hook blocked turn: {e:#}"))?;
            ui.status(&report);
        } else if hooks.join("pre-turn").is_file() {
            ui.status("project hooks skipped: workspace untrusted (run /trust on to enable)");
        }
        self.run_turn_core(input, ui, engine_lease).await
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
        let event_turn = if turn_limit {
            self.turn_count.saturating_add(1)
        } else {
            self.turn_count
        };

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

        let (event_kind, state, verb) = match result {
            Ok(outcome) => match outcome.status {
                TurnStatus::Completed => (
                    EventKind::RunCompleted,
                    ActivityState::Succeeded,
                    ActivityVerb::Complete,
                ),
                TurnStatus::Cancelled => (
                    EventKind::RunCancelled,
                    ActivityState::Cancelled,
                    ActivityVerb::Cancel,
                ),
                TurnStatus::Failed => (
                    EventKind::RunFailed,
                    ActivityState::Failed,
                    ActivityVerb::Fail,
                ),
                TurnStatus::Blocked => (
                    EventKind::RunCompleted,
                    ActivityState::Failed,
                    ActivityVerb::Complete,
                ),
            },
            Err(_) => (
                EventKind::RunFailed,
                ActivityState::Failed,
                ActivityVerb::Fail,
            ),
        };
        let mut run_event = RunEvent::new(
            event_kind,
            EventContext::default(),
            SemanticActivity {
                verb,
                object: ActivityObject::Run,
                state,
                group_key: format!("run:turn:{event_turn}"),
                title: if result.is_ok() {
                    "Run finished"
                } else {
                    "Run failed"
                }
                .into(),
                detail: None,
                refs: Vec::new(),
                progress: None,
            },
        );
        if let Ok(outcome) = result {
            run_event = run_event.with_field("status", serde_json::json!(outcome.status));
            run_event = run_event.with_field("stop_reason", serde_json::json!(outcome.stop_reason));
            if !outcome.changed_files.is_empty() {
                ui.semantic_event(hi_events::RunEvent::new(
                    hi_events::EventKind::GitChanged,
                    hi_events::EventContext::default(),
                    hi_events::SemanticActivity {
                        verb: hi_events::ActivityVerb::Change,
                        object: hi_events::ActivityObject::Git,
                        state: hi_events::ActivityState::Succeeded,
                        group_key: format!("workspace:turn:{event_turn}"),
                        title: "workspace changed".into(),
                        detail: Some(format!("{} file(s) changed", outcome.changed_files.len())),
                        refs: Vec::new(),
                        progress: None,
                    },
                ));
            }
        }
        ui.semantic_event(run_event);

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
        if hooks.join("post-turn").is_file() && hooks_trusted {
            match run_hook_cancellable(self.workspace_root(), "post-turn", &summary, cancellation)
                .await
            {
                Some(Ok(report)) => ui.status(&report),
                Some(Err(error)) => ui.status(&format!("post-turn hook failed: {error:#}")),
                None => return,
            }
        }
        if hooks.join("stop").is_file() && hooks_trusted {
            match run_hook_cancellable(self.workspace_root(), "stop", &summary, cancellation).await
            {
                Some(Ok(report)) => ui.status(&report),
                Some(Err(error)) => ui.status(&format!("stop hook failed: {error:#}")),
                None => (),
            }
        }
    }
}

async fn wait_for_turn_cancellation(cancellation: crate::TurnCancellation) {
    // Bound wakeups: 5ms is snappy enough for interactive Esc/Ctrl+C without a
    // Notify-based redesign of TurnCancellation (still an AtomicBool).
    while !cancellation.is_cancelled() {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

/// Run a project hook until it completes or whole-turn cancellation arrives.
/// `run_hook` owns a private process group, so dropping the losing hook future
/// terminates the hook and its descendants instead of leaving work behind.
async fn run_hook_cancellable(
    workspace: &std::path::Path,
    name: &str,
    input: &str,
    cancellation: &crate::TurnCancellation,
) -> Option<Result<String>> {
    if cancellation.is_cancelled() {
        return None;
    }
    tokio::select! {
        biased;
        _ = wait_for_turn_cancellation(cancellation.clone()) => None,
        result = crate::run_hook(workspace, name, input) => Some(result),
    }
}
