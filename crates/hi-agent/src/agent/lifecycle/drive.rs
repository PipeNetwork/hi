//! Plan/goal auto-drive state and interactive permission transitions.

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
        let stop = outcome
            .or(self.report.last_turn_outcome.as_ref())
            .map(|outcome| outcome.stop_reason);
        crate::PlanDriveAction::decide(
            self.goals.plan_incomplete(),
            self.plan_mode,
            self.plan_drive_paused,
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
        let stop = outcome
            .or(self.report.last_turn_outcome.as_ref())
            .map(|outcome| outcome.stop_reason);
        match stop {
            // A session TurnLimit cannot be advanced by another synthetic
            // turn. Treat it as the same terminal drive stop as cancellation;
            // StepLimit remains resumable and intentionally falls through.
            Some(crate::TurnStopReason::Cancelled | crate::TurnStopReason::TurnLimit) => {
                return crate::DriveAction::Idle {
                    reason: crate::DriveIdleReason::Cancelled,
                };
            }
            Some(crate::TurnStopReason::InfrastructureFailure) => {
                return crate::DriveAction::Idle {
                    reason: crate::DriveIdleReason::Infrastructure,
                };
            }
            _ => {}
        }
        if self.plan_mode {
            return crate::DriveAction::Idle {
                reason: crate::DriveIdleReason::PlanMode,
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
        crate::DriveAction::from_plan(self.plan_drive_decision(outcome))
    }

    /// Whether `/plan pause` has stopped auto-enqueue. The checklist stays pinned.
    pub fn plan_drive_paused(&self) -> bool {
        self.plan_drive_paused
    }

    pub fn plan_drive_stall(&self) -> u32 {
        self.plan_drive_stall
    }

    pub fn plan_drive_status(&self) -> &'static str {
        crate::plan_drive_status(
            self.goals.plan_incomplete(),
            self.plan_drive_paused,
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
        if self.plan_drive_paused == paused {
            return;
        }
        self.plan_drive_paused = paused;
        self.persist_plan_drive();
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
        if self.plan_drive_stall == 0 {
            return;
        }
        self.plan_drive_stall = 0;
        self.persist_plan_drive();
    }

    pub fn restore_plan_drive(&mut self, paused: bool, stall: u32) {
        self.plan_drive_paused = paused;
        self.plan_drive_stall = stall;
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
            return crate::GoalDriveProgress::Stalled { stall: next };
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

    pub(crate) fn maybe_requeue_goal_second_pass(&mut self) -> Option<usize> {
        let mut count = 0;
        let _ = self.update_structured_goal(|goal| {
            count = goal.maybe_requeue_stall_skips();
        });
        if count == 0 {
            return None;
        }
        self.goal_requeue_notice = Some(count);
        Some(count)
    }

    pub fn take_goal_requeue_notice(&mut self) -> Option<usize> {
        self.goal_requeue_notice.take()
    }

    pub fn reset_goal_drive_stall(&mut self) {
        if self.goal_drive_stall == 0 {
            return;
        }
        self.goal_drive_stall = 0;
        self.persist_goal_drive();
    }

    pub fn restore_goal_drive(&mut self, stall: u32) {
        self.goal_drive_stall = stall;
    }

    pub fn clear_pinned_plan(&mut self) {
        self.goals.clear_plan();
        self.plan_drive_paused = false;
        self.plan_drive_stall = 0;
        if let Some(session) = self.session.as_mut() {
            let _ = session.clear_plan();
            let _ = session.record_plan_drive(false, 0);
        }
    }

    fn persist_plan_drive(&mut self) {
        if let Some(session) = self.session.as_mut() {
            let _ = session.record_plan_drive(self.plan_drive_paused, self.plan_drive_stall);
        }
    }

    fn persist_goal_drive(&mut self) {
        if let Some(session) = self.session.as_mut() {
            let _ = session.record_goal_drive(self.goal_drive_stall);
        }
    }

    /// Book-keep pause, stall, and ask_user streak for this turn's prompt.
    pub(crate) fn begin_drive_turn(&mut self, kind: crate::DriveKind) {
        self.finish_drive_turn();
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
                if self.plan_drive_paused || self.plan_drive_stall >= crate::PLAN_DRIVE_STALL_LIMIT
                {
                    self.plan_drive_paused = false;
                    self.plan_drive_stall = 0;
                    self.persist_plan_drive();
                }
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
