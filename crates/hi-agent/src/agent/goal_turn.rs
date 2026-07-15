//! Goal-level turn-end hooks: the long-horizon driver (`goal_turn_end`)
//! that advances/retries the active sub-goal, and `handle_record_decision`
//! for the `record_decision` tool.

use crate::Ui;
use crate::agent::skeptic::SkepticVerdict;
use crate::decision::Decision;
use crate::goal::{DEFAULT_SUBGOAL_RETRIES, Goal, GoalStatus, SkepticStatus};

pub(crate) struct GoalTurnState<'a> {
    pub(crate) stalled_unfinished: bool,
    pub(crate) stalled_repeating: bool,
    pub(crate) hit_step_cap: bool,
    pub(crate) plan_updated_goal: bool,
    pub(crate) proposed_goal: Option<Goal>,
    pub(crate) goal_before: Option<Goal>,
    pub(crate) verified_at: Option<&'a (u64, String)>,
    pub(crate) turn_ledger_revision: u64,
}

impl crate::Agent {
    pub(crate) fn goal_continuation_context(&self, input: &str) -> Option<String> {
        if input != crate::GOAL_CONTINUE_PROMPT {
            return None;
        }
        let goal = self
            .structured_goal
            .as_ref()
            .filter(|goal| goal.should_auto_drive())?;
        let active = goal.active_sub_goal()?;
        let notes = if active.notes.is_empty() {
            String::new()
        } else {
            format!("\nPrior failed attempts:\n- {}", active.notes.join("\n- "))
        };
        Some(format!(
            "Continue the long-horizon goal.\nObjective: {}\nActive sub-goal: {}{}\nComplete this milestone with concrete work and current-revision validation. Preserve the full goal checklist when calling update_plan and append any newly discovered implementation steps.",
            goal.objective, active.description, notes
        ))
    }

    /// Long-horizon driver — called at turn end. When a structured goal is set
    /// and `long_horizon` is on, advance or retry the active sub-goal based on
    /// the turn's outcome, so the next turn resumes at the right sub-goal (with
    /// prior-attempt notes if it stalled, so the model doesn't repeat a failed
    /// approach). The verify retry itself happens *within* the turn (the 'turn
    /// loop re-runs the model on a verify failure); this hook handles the
    /// goal-level progression once the turn settles.
    pub(crate) async fn goal_turn_end(
        &mut self,
        state: GoalTurnState<'_>,
        ui: &mut dyn Ui,
    ) -> bool {
        let GoalTurnState {
            stalled_unfinished,
            stalled_repeating,
            hit_step_cap,
            plan_updated_goal,
            mut proposed_goal,
            goal_before,
            verified_at,
            turn_ledger_revision,
        } = state;
        if !self.config.long_horizon {
            return false;
        }
        let Some(start_goal) = goal_before.as_ref() else {
            return false;
        };
        if start_goal.paused || start_goal.status != GoalStatus::Active {
            return false;
        }
        let start_active_index = start_goal.active_index();
        let mut verification_invalidated = false;
        let max_retries = DEFAULT_SUBGOAL_RETRIES;
        // Only verifier-backed success may advance a long-horizon goal.
        let verified_clean = matches!(self.last_verify, Some(true));
        let mut clean_success =
            verified_clean && !stalled_unfinished && !stalled_repeating && !hit_step_cap;

        // Skeptic gate: on a clean-success turn, a second model reviews the work
        // before its progress stands. It reviews the sub-goal that was active AT
        // TURN START — because `update_plan` may have marked that sub-goal (or the
        // whole goal) done mid-turn, and the model's own "done" claim is exactly
        // what a skeptic should second-guess. On an objection we revert the turn's
        // goal progress (restore the pre-turn goal) and record the objections as a
        // retry note; the edits stay on disk for the next turn to build on.
        // Fail-open — any reviewer error/timeout/unparseable reply approves.
        if clean_success
            && let Some((objective, sub_goal)) = goal_before.as_ref().and_then(|g| {
                if !g.team || g.paused || g.status != GoalStatus::Active {
                    return None;
                }
                let sg = g.active_sub_goal()?;
                Some((g.objective.clone(), sg.description.clone()))
            })
        {
            match self.skeptic_gate(&objective, &sub_goal).await {
                SkepticVerdict::Object(items) => {
                    let objections = items.join("\n");
                    // Objection: revert the turn's goal progress and record it.
                    self.structured_goal = goal_before;
                    if let Some(goal) = self.structured_goal.as_mut() {
                        goal.skeptic_objections = goal.skeptic_objections.saturating_add(1);
                        goal.last_skeptic_status = Some(SkepticStatus::Objected);
                        goal.record_failure(
                            format!("reviewer objected — address then continue:\n{objections}"),
                            max_retries,
                        );
                    }
                    let first = objections.lines().next().unwrap_or("see notes");
                    ui.status(&format!("🔍 skeptic objected — retrying: {first}"));
                    self.refresh_system_message();
                    self.persist_goal(ui);
                    self.last_turn_telemetry.skeptic_last_status = Some(SkepticStatus::Objected);
                    return false;
                }
                SkepticVerdict::Approve => {
                    if let Some(goal) = self.structured_goal.as_mut() {
                        goal.last_skeptic_status = Some(SkepticStatus::Approved);
                    }
                    self.last_turn_telemetry.skeptic_last_status = Some(SkepticStatus::Approved);
                    ui.status("🔍 skeptic approved — advancing");
                }
                SkepticVerdict::Unavailable(reason) => {
                    if let Some(goal) = self.structured_goal.as_mut() {
                        goal.skeptic_unavailable = goal.skeptic_unavailable.saturating_add(1);
                        goal.last_skeptic_status = Some(SkepticStatus::Unavailable);
                    }
                    self.last_turn_telemetry.skeptic_unavailable_count = self
                        .last_turn_telemetry
                        .skeptic_unavailable_count
                        .saturating_add(1);
                    self.last_turn_telemetry.skeptic_last_status = Some(SkepticStatus::Unavailable);
                    ui.status(&format!(
                        "⚠ skeptic unavailable — advancing without review: {reason}"
                    ));
                }
            }
        }

        // The skeptic is an asynchronous model call. Reconcile again before
        // allowing it to advance the goal so edits made while it was reviewing
        // cannot inherit the earlier deterministic pass.
        if clean_success {
            // Keep reconciliation and the revision read under one guard. A
            // chained `ledger().reconcile().map(...)` retains its temporary
            // guard through the map closure, so locking again there would
            // deadlock on this non-reentrant mutex.
            let current = {
                let mut ledger = self.runtime.ledger();
                ledger.reconcile().map(|_| {
                    (
                        ledger.revision(),
                        ledger.workspace_revision(),
                        ledger.changes_since(turn_ledger_revision),
                    )
                })
            };
            match current {
                Ok((revision, digest, changes)) => {
                    let current_pass = verified_at.is_some_and(|(verified_revision, verified)| {
                        *verified_revision == revision && verified == &digest
                    });
                    self.last_changed_files =
                        changes.iter().map(|change| change.path.clone()).collect();
                    self.last_file_changes = changes;
                    if !current_pass {
                        self.last_verify = None;
                        clean_success = false;
                        verification_invalidated = true;
                        self.structured_goal = goal_before.clone();
                        self.refresh_system_message();
                        ui.status(
                            "workspace changed while completion review was running; goal progress was not advanced",
                        );
                    }
                }
                Err(error) => {
                    self.last_verify = None;
                    clean_success = false;
                    verification_invalidated = true;
                    self.structured_goal = goal_before.clone();
                    self.refresh_system_message();
                    ui.status(&format!(
                        "could not confirm the reviewed workspace revision; goal progress was not advanced: {error:#}"
                    ));
                }
            }
        }

        // Only now may a model-authored update_plan become live. Until this
        // point it is turn-local, so every failed/unverified/error exit leaves
        // the durable goal at its pre-turn state.
        if clean_success && let Some(mut proposal) = proposed_goal.take() {
            // The proposal was cloned before the asynchronous skeptic call.
            // Preserve review metadata accumulated on the live baseline goal.
            if let Some(reviewed) = self.structured_goal.as_ref() {
                proposal.skeptic_objections = reviewed.skeptic_objections;
                proposal.skeptic_unavailable = reviewed.skeptic_unavailable;
                proposal.last_skeptic_status = reviewed.last_skeptic_status;
            }
            self.structured_goal = Some(proposal);
        }
        if !clean_success {
            // Defensive restoration for future mutations of the live goal
            // inside a turn. Today update_plan remains entirely provisional,
            // but the failure path stays explicitly anchored to the pre-turn
            // goal before any neutral/failure return.
            self.structured_goal = goal_before.clone();
        }
        // A clean read-only turn (investigation, Q&A — no edits, no verify,
        // no stall) is neutral: neither advance nor record failure. The sub-goal
        // stays active for the next turn, which should do the actual work.
        let no_edit_neutral = self.last_verify.is_none()
            && !stalled_unfinished
            && !stalled_repeating
            && !hit_step_cap
            && self.last_changed_files.is_empty();
        if no_edit_neutral {
            return verification_invalidated;
        }
        if clean_success {
            // Approve (or gate off): advance as today. If `update_plan` already
            // advanced the goal this turn, don't advance again (skips a sub-goal).
            if !plan_updated_goal && let Some(goal) = self.structured_goal.as_mut() {
                let i = goal.active_index();
                goal.advance();
                if let Some(i) = i {
                    ui.status(&format!(
                        "✓ sub-goal {}/{} done — advancing",
                        i + 1,
                        goal.sub_goals.len().max(i + 1)
                    ));
                }
            }
            // A goal about to finish must first pass the completion audit —
            // one bounded side-call comparing the "done" claim against the
            // objective's referenced documents and the real repository. It can
            // only hold the goal open (append missing work), never advance it,
            // so no post-skeptic reconcile pass is needed here. Fail-open.
            if matches!(
                self.structured_goal.as_ref().map(|g| g.status),
                Some(GoalStatus::Done)
            ) {
                self.audit_goal_completion(ui).await;
            }
            if matches!(
                self.structured_goal.as_ref().map(|g| g.status),
                Some(GoalStatus::Done)
            ) {
                ui.status("✓ long-horizon goal complete");
            } else if plan_updated_goal
                && let (Some(before), Some(after)) = (
                    start_active_index,
                    self.structured_goal.as_ref().and_then(Goal::active_index),
                )
                && after > before
                && let Some(goal) = self.structured_goal.as_ref()
            {
                ui.status(&format!(
                    "✓ sub-goal {}/{} done — advancing",
                    before + 1,
                    goal.sub_goals.len().max(before + 1)
                ));
            }
            self.refresh_system_message();
            self.persist_goal(ui);
            return verification_invalidated;
        }
        // A stalled or cap-hit turn, or a verify failure that ended the turn,
        // records a sub-goal attempt so the next turn sees the prior note. If
        // the budget is exhausted, the sub-goal (and goal) is marked Failed.
        let reason = if hit_step_cap {
            "hit the per-turn step cap"
        } else if self.last_verify == Some(false) {
            "verification failed and the turn ended without fixing it"
        } else if stalled_repeating {
            "stalled repeating the same tool call"
        } else if stalled_unfinished {
            "ended without completing the requested work"
        } else if self.last_verify.is_none() && !self.last_changed_files.is_empty() {
            "ended with unverified workspace changes"
        } else {
            "verification failed and the turn ended without fixing it"
        };
        let can_retry = match self.structured_goal.as_mut() {
            Some(goal) => goal.record_failure(reason, max_retries),
            None => return verification_invalidated,
        };
        if can_retry {
            ui.status(&format!(
                "↻ sub-goal failed this turn ({reason}) — will retry next turn, don't repeat the same approach"
            ));
        } else {
            // Budget exhausted. When drivable work remains, skip past the
            // failed step instead of failing the whole goal — one stuck
            // milestone must not kill a mostly-done run (the step stays
            // `Failed` and visible). Only a dead end with nothing left to
            // drive is terminal.
            let skipped = self
                .structured_goal
                .as_mut()
                .is_some_and(Goal::continue_past_failure);
            if skipped {
                ui.status(&format!(
                    "✗ sub-goal exhausted its retry budget ({reason}) — marked failed, skipping to the next step; /goal to revisit"
                ));
            } else {
                ui.status(&format!(
                    "✗ sub-goal exhausted its retry budget ({reason}) — marked failed; /goal to revise or continue past it"
                ));
            }
        }
        self.refresh_system_message();
        self.persist_goal(ui);
        verification_invalidated
    }

    /// Handle a `record_decision` tool call: parse the args, append to the
    /// durable decision log (which feeds the system prompt), and return a
    /// terse confirmation for the model. Malformed args yield an error string
    /// (the model sees it and can retry), not a panic.
    pub(crate) fn handle_record_decision(&mut self, arguments: &str) -> hi_tools::ToolOutcome {
        #[derive(serde::Deserialize)]
        struct DecisionArgs {
            summary: String,
            rationale: String,
            #[serde(default)]
            files: Vec<String>,
        }
        match serde_json::from_str::<DecisionArgs>(arguments) {
            Ok(args) => {
                let summary = args.summary.trim().to_string();
                if summary.is_empty() {
                    return decision_tool_outcome(
                        "Error: record_decision needs a non-empty summary".to_string(),
                        hi_tools::ToolStatus::Failed,
                    );
                }
                let mut next = self.decisions.clone();
                next.record(Decision {
                    summary,
                    rationale: args.rationale.trim().to_string(),
                    files: args.files,
                });
                if let Some(session) = self.session.as_mut()
                    && let Err(err) = session.record_decisions(&next)
                {
                    return decision_tool_outcome(
                        format!("Error: couldn't persist decision: {err}"),
                        hi_tools::ToolStatus::Failed,
                    );
                }
                self.decisions = next;
                // Refresh the system prompt so the decision is injected on the
                // next turn (and visible to the model immediately in history).
                self.refresh_system_message();
                decision_tool_outcome(
                    "Decision recorded — it will persist across compaction.".to_string(),
                    hi_tools::ToolStatus::Succeeded,
                )
            }
            Err(err) => decision_tool_outcome(
                format!("Error: bad record_decision arguments: {err}"),
                hi_tools::ToolStatus::Failed,
            ),
        }
    }
}

fn decision_tool_outcome(content: String, status: hi_tools::ToolStatus) -> hi_tools::ToolOutcome {
    hi_tools::ToolOutcome {
        content,
        display: None,
        plan: None,
        status,
        process: None,
        background: None,
        effects: hi_tools::ToolEffects::default(),
        truncation: hi_tools::TruncationState::Complete,
    }
}
