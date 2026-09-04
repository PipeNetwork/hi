//! Plan/goal auto-drive state and interactive permission transitions.

use anyhow::Result;

impl crate::Agent {
    /// Install an unfinished plan reconstructed by session storage.
    pub fn restore_plan(&mut self, plan: Vec<hi_tools::PlanStep>) {
        self.goals.set_plan_if_pending(plan);
    }

    pub fn current_plan(&self) -> &[hi_tools::PlanStep] {
        self.goals.plan()
    }

    /// Whether the pinned checklist still has pending or active steps.
    pub fn plan_incomplete(&self) -> bool {
        self.goals.plan_incomplete()
    }

    /// Leftover work the next drive would actually run (goal if auto-driving,
    /// else plan).
    pub fn leftover_work(&self) -> Option<String> {
        self.goals.leftover_work()
    }

    /// Checklist leftover only, even when a structured goal would shadow it.
    pub fn plan_leftover_work(&self) -> Option<String> {
        self.goals.plan_leftover_work()
    }

    /// Title of the first active, else pending, **checklist** step.
    pub fn next_plan_step_title(&self) -> Option<&str> {
        self.goals.next_checklist_step_title()
    }

    /// Canonical leftover-work gate. Frontends must not reimplement this.
    pub fn plan_drive_decision(
        &self,
        outcome: Option<&crate::TurnOutcome>,
    ) -> crate::PlanDriveAction {
        self.plan_drive_decision_for_outcome(outcome.or(self.report.last_turn_outcome.as_ref()))
    }

    fn plan_drive_decision_for_outcome(
        &self,
        outcome: Option<&crate::TurnOutcome>,
    ) -> crate::PlanDriveAction {
        if self.goals.plan_incomplete() && self.plan_approval_parked {
            return crate::PlanDriveAction::Idle {
                reason: crate::PlanDriveIdleReason::ApprovalParked,
            };
        }
        match outcome.map(|outcome| (outcome.status, outcome.stop_reason)) {
            Some((crate::TurnStatus::Cancelled, _))
            | Some((_, crate::TurnStopReason::Cancelled | crate::TurnStopReason::TurnLimit)) => {
                return crate::PlanDriveAction::Idle {
                    reason: crate::PlanDriveIdleReason::Cancelled,
                };
            }
            Some((_, crate::TurnStopReason::InfrastructureFailure)) => {
                return crate::PlanDriveAction::Idle {
                    reason: crate::PlanDriveIdleReason::Infrastructure,
                };
            }
            Some((_, crate::TurnStopReason::NoProgress)) => {
                return crate::PlanDriveAction::Idle {
                    reason: crate::PlanDriveIdleReason::NoProgress,
                };
            }
            Some((crate::TurnStatus::Blocked, _)) => {
                return crate::PlanDriveAction::Idle {
                    reason: crate::PlanDriveIdleReason::Blocked,
                };
            }
            _ => {}
        }
        if outcome.is_some_and(crate::plan_drive::outcome_blocks_automatic_drive) {
            return crate::PlanDriveAction::Idle {
                reason: crate::PlanDriveIdleReason::Blocked,
            };
        }
        let stop = outcome.map(|outcome| outcome.stop_reason);
        crate::PlanDriveAction::decide(
            self.goals.plan_incomplete(),
            self.plan_mode,
            self.plan_drive_paused(),
            self.plan_drive_stall,
            self.goals
                .structured
                .as_ref()
                .is_some_and(crate::goal::Goal::should_auto_drive),
            stop,
        )
    }

    /// Unified leftover-work gate: goal drive wins over plan when both apply.
    pub fn drive_decision(&self, outcome: Option<&crate::TurnOutcome>) -> crate::DriveAction {
        self.drive_decision_for_outcome(outcome.or(self.report.last_turn_outcome.as_ref()))
    }

    /// Drive gate after an explicit user command installs or resumes a goal.
    /// This deliberately ignores the previous turn's terminal outcome; only
    /// automatic post-turn/startup drive is latched by failure or cancellation.
    pub fn explicit_goal_drive_decision(&self) -> crate::DriveAction {
        self.drive_decision_for_outcome(None)
    }

    fn drive_decision_for_outcome(
        &self,
        outcome: Option<&crate::TurnOutcome>,
    ) -> crate::DriveAction {
        match outcome.map(|outcome| (outcome.status, outcome.stop_reason)) {
            Some((crate::TurnStatus::Cancelled, _))
            | Some((_, crate::TurnStopReason::Cancelled | crate::TurnStopReason::TurnLimit)) => {
                return crate::DriveAction::Idle {
                    reason: crate::DriveIdleReason::Cancelled,
                };
            }
            Some((_, crate::TurnStopReason::InfrastructureFailure)) => {
                return crate::DriveAction::Idle {
                    reason: crate::DriveIdleReason::Infrastructure,
                };
            }
            Some((_, crate::TurnStopReason::NoProgress)) => {
                return crate::DriveAction::Idle {
                    reason: crate::DriveIdleReason::NoProgress,
                };
            }
            Some((crate::TurnStatus::Blocked, _)) => {
                return crate::DriveAction::Idle {
                    reason: crate::DriveIdleReason::Blocked,
                };
            }
            _ => {}
        }
        if outcome.is_some_and(crate::plan_drive::outcome_blocks_automatic_drive) {
            return crate::DriveAction::Idle {
                reason: crate::DriveIdleReason::Blocked,
            };
        }
        if self.plan_mode {
            return crate::DriveAction::Idle {
                reason: crate::DriveIdleReason::PlanMode,
            };
        }
        // Approval belongs to the proposed work, so a structured goal must not
        // bypass it merely because goal driving normally wins over checklists.
        if self.goals.plan_incomplete() && self.plan_approval_parked {
            return crate::DriveAction::Idle {
                reason: crate::DriveIdleReason::PlanApprovalParked,
            };
        }
        if self
            .goals
            .structured
            .as_ref()
            .is_some_and(crate::goal::Goal::has_drive_work)
        {
            if self
                .goals
                .structured
                .as_ref()
                .is_some_and(crate::goal::Goal::is_paused)
            {
                return crate::DriveAction::Idle {
                    reason: crate::DriveIdleReason::GoalPaused,
                };
            }
            if self.goal_drive_stall >= crate::GOAL_DRIVE_STALL_LIMIT {
                return crate::DriveAction::Idle {
                    reason: crate::DriveIdleReason::GoalParked,
                };
            }
            return crate::DriveAction::Enqueue(crate::DriveKind::Goal);
        }
        crate::DriveAction::from_plan(self.plan_drive_decision_for_outcome(outcome))
    }

    /// Whether `/plan pause` has stopped auto-enqueue. The checklist stays pinned.
    pub fn plan_drive_paused(&self) -> bool {
        self.plan_drive_pause.is_paused()
            && !self.pending_plan_interruption_resume
            && !self.turn_consumed_plan_interruption
    }

    pub(crate) fn durable_plan_drive_paused(&self) -> bool {
        self.plan_drive_pause.is_paused()
    }

    pub(crate) fn plan_drive_resumes_on_user_input(&self) -> bool {
        self.plan_drive_pause.resumes_on_user_input()
    }

    /// Whether proposed plan work still requires approval. The durable legacy
    /// name covers drafts, revisions, and visible or parked approval cards.
    pub fn plan_approval_parked(&self) -> bool {
        self.plan_approval_parked
    }

    /// Set the pending approval gate independently from `/plan pause`. Opening
    /// or reopening a card keeps this gate set until approval is accepted.
    pub fn set_plan_approval_parked(&mut self, parked: bool) {
        let _ = self.try_set_plan_approval_parked(parked);
    }

    /// Persist an approval transition before publishing it to the live agent.
    /// A failed approval write must never release autonomous execution.
    pub fn try_set_plan_approval_parked(&mut self, parked: bool) -> Result<bool> {
        if self.plan_approval_parked == parked {
            return Ok(false);
        }
        if let Some(session) = self.session.as_mut() {
            session.record_plan_approval_parked(parked)?;
        }
        self.plan_approval_parked = parked;
        Ok(true)
    }

    pub fn plan_drive_stall(&self) -> u32 {
        self.plan_drive_stall
    }

    pub fn plan_drive_status(&self) -> &'static str {
        crate::plan_drive_status(
            self.goals.plan_incomplete(),
            self.plan_drive_paused(),
            self.plan_drive_stall,
        )
    }

    pub fn goal_drive_stall(&self) -> u32 {
        self.goal_drive_stall
    }

    pub fn goal_drive_status(&self) -> &'static str {
        crate::goal_drive_status(
            self.goals
                .structured
                .as_ref()
                .is_some_and(crate::goal::Goal::has_drive_work),
            self.goals
                .structured
                .as_ref()
                .is_some_and(crate::goal::Goal::is_paused),
            self.goal_drive_stall,
        )
    }

    pub fn set_interactive_session(&mut self, interactive: bool) {
        self.interactive_session = interactive;
    }

    pub fn set_plan_drive_paused(&mut self, paused: bool) {
        let _ = self.try_set_plan_drive_paused(paused);
    }

    /// Persist an explicit manual pause/unpause, reverting the live state when
    /// the session sink rejects the transition.
    pub fn try_set_plan_drive_paused(&mut self, paused: bool) -> Result<bool> {
        let pause = if paused {
            crate::plan_drive::PlanDrivePause::Manual
        } else {
            crate::plan_drive::PlanDrivePause::Running
        };
        if self.plan_drive_pause == pause
            && !self.pending_plan_interruption_resume
            && !self.turn_consumed_plan_interruption
        {
            return Ok(false);
        }
        let previous_pause = self.plan_drive_pause;
        let previous_pending = self.pending_plan_interruption_resume;
        let previous_consumed = self.turn_consumed_plan_interruption;
        self.plan_drive_pause = pause;
        self.pending_plan_interruption_resume = false;
        self.turn_consumed_plan_interruption = false;
        if let Err(error) = self.try_persist_plan_drive_evidence_delta(false, &[]) {
            self.plan_drive_pause = previous_pause;
            self.pending_plan_interruption_resume = previous_pending;
            self.turn_consumed_plan_interruption = previous_consumed;
            return Err(error);
        }
        Ok(true)
    }

    /// Resume explicit pause/park state in one durable record so restart can
    /// never observe `paused=false` with the old parked stall ledger.
    pub fn resume_plan_drive(&mut self) -> Result<bool> {
        let previous_pause = self.plan_drive_pause;
        let previous_stall = self.plan_drive_stall;
        let previous_evidence = self.plan_drive_evidence.clone();
        let previous_pending = self.pending_plan_interruption_resume;
        let previous_consumed = self.turn_consumed_plan_interruption;
        let reset_evidence = !self.plan_drive_evidence.is_empty();
        let changed = self.durable_plan_drive_paused()
            || self.plan_drive_stall != 0
            || reset_evidence
            || previous_pending
            || previous_consumed;
        if !changed {
            return Ok(false);
        }
        self.plan_drive_pause = crate::plan_drive::PlanDrivePause::Running;
        self.plan_drive_stall = 0;
        self.plan_drive_evidence.clear();
        self.pending_plan_interruption_resume = false;
        self.turn_consumed_plan_interruption = false;
        if let Err(error) = self.try_persist_plan_drive_evidence_delta(reset_evidence, &[]) {
            self.plan_drive_pause = previous_pause;
            self.plan_drive_stall = previous_stall;
            self.plan_drive_evidence = previous_evidence;
            self.pending_plan_interruption_resume = previous_pending;
            self.turn_consumed_plan_interruption = previous_consumed;
            return Err(error);
        }
        Ok(true)
    }

    /// Stop an interrupted synthetic plan turn without requiring a special
    /// command to recover. Autonomous drive remains latched off until genuine
    /// user work arrives; `/plan pause` uses the separate manual state.
    pub fn pause_plan_drive_until_user_input(&mut self) -> Result<bool> {
        let was_effectively_paused = self.plan_drive_paused();
        let previous_pending = self.pending_plan_interruption_resume;
        let previous_consumed = self.turn_consumed_plan_interruption;
        self.pending_plan_interruption_resume = false;
        self.turn_consumed_plan_interruption = false;
        if self.plan_drive_pause == crate::plan_drive::PlanDrivePause::Interrupted {
            return Ok(!was_effectively_paused);
        }
        let previous = self.plan_drive_pause;
        self.plan_drive_pause = crate::plan_drive::PlanDrivePause::Interrupted;
        if let Err(error) = self.try_persist_plan_drive_evidence_delta(false, &[]) {
            self.plan_drive_pause = previous;
            self.pending_plan_interruption_resume = previous_pending;
            self.turn_consumed_plan_interruption = previous_consumed;
            return Err(error);
        }
        Ok(true)
    }

    /// Apply the prompt-owned plan-drive transition before a turn is rendered
    /// or executed. Returns true when a visible pause or park was consumed.
    pub fn prepare_plan_drive_for_turn(&mut self, kind: crate::DriveKind) -> Result<bool> {
        if kind == crate::DriveKind::User
            && self.plan_drive_pause.resumes_on_user_input()
            && !self.pending_plan_interruption_resume
            && !self.turn_consumed_plan_interruption
        {
            // Keep the durable latch until this steering turn completes. The
            // transient flag makes the active turn render as resumed while a
            // crash, cancellation, or early error still restores as paused.
            self.pending_plan_interruption_resume = true;
            return Ok(true);
        }
        let resume = match kind {
            // A synthetic plan prompt is an explicit resume only when drive is
            // actually paused or parked. Ordinary automatic plan turns must
            // retain their cross-turn stall/evidence ledger.
            crate::DriveKind::Plan => {
                self.plan_drive_paused() || self.plan_drive_stall >= crate::PLAN_DRIVE_STALL_LIMIT
            }
            crate::DriveKind::User => false,
            crate::DriveKind::Goal => false,
        };
        if !resume {
            return Ok(false);
        }
        let previous_pause = self.plan_drive_pause;
        let previous_stall = self.plan_drive_stall;
        let previous_evidence = self.plan_drive_evidence.clone();
        let was_paused = self.plan_drive_paused();
        let reset_evidence = self.plan_drive_evidence.clear();
        let changed = was_paused || self.plan_drive_stall != 0 || reset_evidence;
        self.plan_drive_pause = crate::plan_drive::PlanDrivePause::Running;
        self.plan_drive_stall = 0;
        if changed
            && let Err(error) = self.try_persist_plan_drive_evidence_delta(reset_evidence, &[])
        {
            self.plan_drive_pause = previous_pause;
            self.plan_drive_stall = previous_stall;
            self.plan_drive_evidence = previous_evidence;
            return Err(error);
        }
        Ok(changed)
    }

    /// Commit or roll back a transactional interruption resume after the user
    /// turn settles. Until successful completion, the durable state remains
    /// Interrupted so a crash/restart cannot autonomously drive the plan.
    pub(crate) fn settle_plan_interruption_resume(&mut self, successful: bool) -> Result<()> {
        let pending = std::mem::take(&mut self.pending_plan_interruption_resume);
        let active = std::mem::take(&mut self.turn_consumed_plan_interruption);
        let consumed = pending || active;
        if !consumed || !successful {
            return Ok(());
        }
        debug_assert_eq!(
            self.plan_drive_pause,
            crate::plan_drive::PlanDrivePause::Interrupted
        );
        self.plan_drive_pause = crate::plan_drive::PlanDrivePause::Running;
        if let Err(error) = self.try_persist_plan_drive_evidence_delta(false, &[]) {
            self.plan_drive_pause = crate::plan_drive::PlanDrivePause::Interrupted;
            return Err(error);
        }
        Ok(())
    }

    /// Judge one completed synthetic plan turn against evidence already credited
    /// in this checklist-step scope. This lives on `Agent` so every frontend uses
    /// the same cross-turn ledger instead of interpreting turn-local telemetry.
    pub fn plan_drive_turn_made_progress(&mut self, before_step: Option<&str>) -> bool {
        let structural_progress = before_step != self.next_plan_step_title();
        let mutation_progress = !self.workspace.last_changed_files.is_empty()
            || self
                .report
                .last_turn_telemetry
                .progress_events
                .iter()
                .any(crate::plan_drive::progress_event_is_structural_or_mutation);
        if structural_progress || mutation_progress {
            self.clear_plan_drive_evidence_scope();
            return true;
        }

        // An implementation step may spend one turn orienting itself, but a
        // stream of novel offsets/files is not durable implementation progress.
        // Remember whether this structural scope already received that initial
        // evidence credit before recording the current turn's signatures. The
        // ledger is persisted, so a restart cannot renew the orientation turn.
        let already_oriented = !self.plan_drive_evidence.is_empty();
        let added = self
            .plan_drive_evidence
            .record_novel(&self.report.last_turn_telemetry);
        if added.is_empty() {
            return false;
        }
        self.persist_plan_drive_evidence_delta(false, &added);
        crate::plan_drive::plan_step_allows_continual_evidence(before_step) || !already_oriented
    }

    pub fn note_plan_drive_progress(&mut self, made_progress: bool) {
        let next = crate::next_plan_drive_stall(true, made_progress, self.plan_drive_stall);
        if next == self.plan_drive_stall {
            return;
        }
        self.plan_drive_stall = next;
        self.persist_plan_drive();
    }

    pub fn reset_plan_drive_stall(&mut self) {
        let reset_evidence = self.plan_drive_evidence.clear();
        if self.plan_drive_stall == 0 && !reset_evidence {
            return;
        }
        self.plan_drive_stall = 0;
        self.persist_plan_drive_evidence_delta(reset_evidence, &[]);
    }

    pub fn restore_plan_drive(&mut self, paused: bool, stall: u32, evidence_hashes: Vec<String>) {
        self.restore_plan_drive_with_policy(paused, false, stall, evidence_hashes);
    }

    /// Restore plan drive state including the interruption-only resume policy.
    /// The legacy [`Self::restore_plan_drive`] entry point retains manual-pause
    /// semantics for downstream callers compiled against the public API.
    pub fn restore_plan_drive_with_policy(
        &mut self,
        paused: bool,
        resume_on_user_input: bool,
        stall: u32,
        evidence_hashes: Vec<String>,
    ) {
        self.plan_drive_pause = if !paused {
            crate::plan_drive::PlanDrivePause::Running
        } else if resume_on_user_input {
            crate::plan_drive::PlanDrivePause::Interrupted
        } else {
            crate::plan_drive::PlanDrivePause::Manual
        };
        self.pending_plan_interruption_resume = false;
        self.turn_consumed_plan_interruption = false;
        self.plan_drive_stall = stall;
        self.plan_drive_evidence.restore(evidence_hashes);
    }

    pub fn restore_plan_approval_parked(&mut self, parked: bool) {
        self.plan_approval_parked = parked;
    }

    pub fn note_goal_drive_progress(&mut self, made_progress: bool) -> crate::GoalDriveProgress {
        if made_progress {
            let reset = self.goal_drive_stall != 0;
            if reset {
                self.goal_drive_stall = 0;
                self.persist_goal_drive();
            }
            if let Some(count) = self.maybe_requeue_goal_second_pass() {
                return crate::GoalDriveProgress::Requeued { count };
            }
            return if reset {
                crate::GoalDriveProgress::Reset
            } else {
                crate::GoalDriveProgress::Unchanged
            };
        }
        let next = crate::next_plan_drive_stall(true, false, self.goal_drive_stall);
        if next < crate::GOAL_DRIVE_STALL_LIMIT {
            if next == self.goal_drive_stall {
                return crate::GoalDriveProgress::Unchanged;
            }
            self.goal_drive_stall = next;
            self.persist_goal_drive();
            return crate::GoalDriveProgress::NoProgress { count: next };
        }
        let Some(goal) = self.goals.structured.as_ref() else {
            self.goal_drive_stall = next;
            self.persist_goal_drive();
            return crate::GoalDriveProgress::Parked;
        };
        if goal.is_thrashing() {
            self.goal_drive_stall = next;
            self.persist_goal_drive();
            return crate::GoalDriveProgress::Parked;
        }
        let has_successor = goal.active_index().is_some_and(|index| {
            goal.sub_goals
                .iter()
                .skip(index + 1)
                .any(|step| step.status == crate::GoalStatus::Pending)
        });
        let failed = goal
            .active_sub_goal()
            .map(|step| step.description.clone())
            .unwrap_or_default();
        if has_successor {
            let _ = self.update_structured_goal(|goal| {
                goal.skip_stalled_active(format!(
                    "skipped after {} drive turns with no progress",
                    crate::GOAL_DRIVE_STALL_LIMIT
                ));
            });
            self.clear_goal_drive_evidence_scope();
            let after = self.goals.structured.as_ref();
            let next_title = after
                .and_then(|goal| goal.active_sub_goal())
                .map(|step| step.description.clone());
            if after.is_some_and(|goal| goal.is_thrashing()) {
                self.goal_drive_stall = next;
                self.persist_goal_drive();
                return crate::GoalDriveProgress::Parked;
            }
            if after.is_none_or(|goal| goal.active_sub_goal().is_none()) {
                if let Some(count) = self.maybe_requeue_goal_second_pass() {
                    self.goal_drive_stall = 0;
                    self.persist_goal_drive();
                    return crate::GoalDriveProgress::Requeued { count };
                }
                self.goal_drive_stall = next;
                self.persist_goal_drive();
                return crate::GoalDriveProgress::Parked;
            }
            self.goal_drive_stall = 0;
            self.persist_goal_drive();
            return crate::GoalDriveProgress::Skipped {
                failed,
                next: next_title,
            };
        }
        // Last remaining step: only skip-and-requeue when something already
        // completed (a second pass is possible). Otherwise park as today so a
        // single stuck step still GoalParked with the cursor intact.
        if self
            .goals
            .structured
            .as_ref()
            .is_some_and(|goal| goal.completed_count() > 0)
        {
            let _ = self.update_structured_goal(|goal| {
                goal.skip_stalled_active(format!(
                    "skipped after {} drive turns with no progress",
                    crate::GOAL_DRIVE_STALL_LIMIT
                ));
            });
            self.clear_goal_drive_evidence_scope();
            if let Some(count) = self.maybe_requeue_goal_second_pass() {
                self.goal_drive_stall = 0;
                self.persist_goal_drive();
                return crate::GoalDriveProgress::Requeued { count };
            }
        }
        self.goal_drive_stall = next;
        self.persist_goal_drive();
        crate::GoalDriveProgress::Parked
    }

    /// Goal-drive counterpart to [`Self::plan_drive_turn_made_progress`]. Goal
    /// retry counters/notes do not clear the ledger; only a real structural
    /// transition, mutation, or previously unseen signed inspection does.
    pub fn goal_drive_turn_made_progress(&mut self, before: Option<&crate::Goal>) -> bool {
        let after = self.goals.structured.as_ref();
        let structural_progress = match (after, before) {
            (Some(after), Some(before)) => after.drive_state_changed_since(before),
            _ => true,
        };
        let mutation_progress = !self.workspace.last_changed_files.is_empty()
            || self
                .report
                .last_turn_telemetry
                .progress_events
                .iter()
                .any(crate::plan_drive::progress_event_is_structural_or_mutation);
        if structural_progress || mutation_progress {
            self.clear_goal_drive_evidence_scope();
            return true;
        }

        let added = self
            .goal_drive_evidence
            .record_novel(&self.report.last_turn_telemetry);
        if added.is_empty() {
            return false;
        }
        self.persist_goal_drive_evidence_delta(false, &added);
        true
    }

    pub(crate) fn maybe_requeue_goal_second_pass(&mut self) -> Option<usize> {
        let mut count = 0;
        let _ = self.update_structured_goal(|goal| {
            count = goal.maybe_requeue_stall_skips();
        });
        if count == 0 {
            return None;
        }
        self.clear_goal_drive_evidence_scope();
        self.goal_requeue_notice = Some(count);
        Some(count)
    }

    pub fn take_goal_requeue_notice(&mut self) -> Option<usize> {
        self.goal_requeue_notice.take()
    }

    pub fn reset_goal_drive_stall(&mut self) {
        let reset_evidence = self.goal_drive_evidence.clear();
        if self.goal_drive_stall == 0 && !reset_evidence {
            return;
        }
        self.goal_drive_stall = 0;
        self.persist_goal_drive_evidence_delta(reset_evidence, &[]);
    }

    pub fn restore_goal_drive(&mut self, stall: u32, evidence_hashes: Vec<String>) {
        self.goal_drive_stall = stall;
        self.goal_drive_evidence.restore(evidence_hashes);
    }

    pub fn clear_pinned_plan(&mut self) {
        self.goals.clear_plan();
        self.plan_drive_pause = crate::plan_drive::PlanDrivePause::Running;
        self.pending_plan_interruption_resume = false;
        self.turn_consumed_plan_interruption = false;
        self.plan_approval_parked = false;
        self.plan_drive_stall = 0;
        self.plan_drive_evidence.clear();
        if let Some(session) = self.session.as_mut() {
            let _ = session.clear_plan();
            let _ = session.record_plan_drive_state_with_policy(false, 0, false, true, &[]);
            let _ = session.record_plan_approval_parked(false);
        }
    }

    fn persist_plan_drive(&mut self) {
        self.persist_plan_drive_evidence_delta(false, &[]);
    }

    fn persist_plan_drive_evidence_delta(&mut self, reset: bool, added: &[String]) {
        let _ = self.try_persist_plan_drive_evidence_delta(reset, added);
    }

    fn try_persist_plan_drive_evidence_delta(
        &mut self,
        reset: bool,
        added: &[String],
    ) -> Result<()> {
        let paused = self.durable_plan_drive_paused();
        let resume_on_user_input = self.plan_drive_resumes_on_user_input();
        if let Some(session) = self.session.as_mut() {
            session.record_plan_drive_state_with_policy(
                paused,
                self.plan_drive_stall,
                resume_on_user_input,
                reset,
                added,
            )?;
        }
        Ok(())
    }

    fn clear_plan_drive_evidence_scope(&mut self) {
        if self.plan_drive_evidence.clear() {
            self.persist_plan_drive_evidence_delta(true, &[]);
        }
    }

    fn persist_goal_drive(&mut self) {
        self.persist_goal_drive_evidence_delta(false, &[]);
    }

    fn persist_goal_drive_evidence_delta(&mut self, reset: bool, added: &[String]) {
        if let Some(session) = self.session.as_mut() {
            let _ = session.record_goal_drive_state(self.goal_drive_stall, reset, added);
        }
    }

    fn clear_goal_drive_evidence_scope(&mut self) {
        if self.goal_drive_evidence.clear() {
            self.persist_goal_drive_evidence_delta(true, &[]);
        }
    }

    /// Book-keep pause, stall, and ask_user streak for this turn's prompt.
    pub(crate) fn begin_drive_turn(&mut self, kind: crate::DriveKind) -> Result<()> {
        self.finish_drive_turn();
        let preflight_consumed_interruption = if kind == crate::DriveKind::User {
            std::mem::take(&mut self.pending_plan_interruption_resume)
        } else {
            self.pending_plan_interruption_resume = false;
            false
        };
        if !preflight_consumed_interruption {
            self.prepare_plan_drive_for_turn(kind)?;
        }
        let prepared_interruption = std::mem::take(&mut self.pending_plan_interruption_resume);
        self.turn_consumed_plan_interruption = preflight_consumed_interruption
            || (kind == crate::DriveKind::User && prepared_interruption);
        self.turn_drive_kind = kind;
        self.ask_user_calls = 0;
        self.approval_parked = false;
        match kind {
            crate::DriveKind::User => {
                self.ask_user_drive_streak = 0;
                self.reset_plan_drive_stall();
                self.reset_goal_drive_stall();
            }
            crate::DriveKind::Plan => {
                self.maybe_demote_always_for_drive();
            }
            crate::DriveKind::Goal => {
                let parked = self.goal_drive_stall >= crate::GOAL_DRIVE_STALL_LIMIT;
                let paused = self
                    .goals
                    .structured
                    .as_ref()
                    .is_some_and(crate::goal::Goal::is_paused);
                if paused {
                    let _ = self.try_set_goal_pause_reason(crate::GoalPauseReason::None);
                }
                if parked || paused {
                    self.reset_goal_drive_stall();
                }
                self.apply_goal_drive_permissions();
            }
        }
        Ok(())
    }

    fn apply_goal_drive_permissions(&mut self) {
        // Unattended is routing (park confirms, notify), not YOLO.
        self.maybe_demote_always_for_drive();
    }

    fn maybe_demote_always_for_drive(&mut self) {
        if !self.interactive_session {
            return;
        }
        if self.permission_mode == crate::PermissionMode::Always {
            self.drive_restore_permission = Some(crate::PermissionMode::Always);
            self.set_permission_mode(crate::PermissionMode::Auto);
        }
    }

    /// Restore Always after an interactive synthetic drive turn.
    pub(crate) fn finish_drive_turn(&mut self) {
        if let Some(mode) = self.drive_restore_permission.take() {
            self.set_permission_mode(mode);
        }
    }

    /// Expand the synthetic plan-drive prompt with the next leftover step.
    pub(crate) fn plan_continuation_context(&self, input: &str) -> Option<String> {
        if crate::DriveKind::from_prompt(input) != crate::DriveKind::Plan {
            return None;
        }
        let leftover = self.goals.plan_leftover_work()?;
        Some(format!(
            "{}\nNext: {leftover}. Use your tools to do that work now.",
            crate::PLAN_DRIVE_PROMPT
        ))
    }

    /// Whether `/plan` mode is active (frontends should prefer read-only tools).
    pub fn plan_mode(&self) -> bool {
        self.plan_mode
    }

    /// Planning is an execution restriction even when a full/minimal catalog
    /// or the permission ladder would otherwise allow workspace mutations.
    pub(crate) fn effective_tool_mode(&self) -> hi_ai::ToolMode {
        if self.plan_mode && self.config.routing.tool_mode != hi_ai::ToolMode::ChatOnly {
            hi_ai::ToolMode::ReadOnly
        } else {
            self.config.routing.tool_mode
        }
    }

    pub fn set_plan_mode(&mut self, on: bool) {
        self.plan_mode = on;
        if on {
            // Plan mode pairs with ask-style caution for accidental mutations.
            if self.permission_mode == crate::PermissionMode::Always {
                self.set_permission_mode(crate::PermissionMode::Ask);
            }
            // Advertise tools as if the next task were read-only (no mutations).
            self.set_advertised_tools(Some(("", crate::TaskIntent::ReadOnly)));
        } else {
            // Mode controls are per-turn transport, not durable user intent.
            // Clean them immediately so `/plan off` also fixes already-loaded
            // legacy sessions before `/compact`, `/recap`, or the next turn.
            self.messages.strip_previous_turn_blocks();
            self.persisted = self.persisted.min(self.messages.len());
            self.set_advertised_tools(None);
        }
    }

    /// Toggle Claude-style post-turn suggested next prompts (`/config suggest`).
    pub fn set_suggest_next_prompt(&mut self, on: bool) {
        self.config.memory.suggest_next_prompt = on;
    }

    pub fn permission_mode(&self) -> crate::PermissionMode {
        self.permission_mode
    }

    pub fn approval_parked(&self) -> bool {
        self.approval_parked
    }

    pub(crate) fn note_approval_parked(&mut self, ui: &mut dyn crate::Ui) {
        self.approval_parked = true;
        ui.status(crate::PARKED_FOR_APPROVAL_STATUS);
        let _ = self.try_set_goal_pause_reason(crate::GoalPauseReason::Approval);
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
}
