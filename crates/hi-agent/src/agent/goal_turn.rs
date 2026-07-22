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
    /// The turn ended because verification itself could not run to a verdict
    /// (timed out, snapshot failed), not because the work was wrong.
    pub(crate) verification_infrastructure_error: bool,
}

impl crate::Agent {
    pub(crate) fn goal_continuation_context(&self, input: &str) -> Option<String> {
        if input != crate::GOAL_CONTINUE_PROMPT {
            return None;
        }
        let goal = self
            .goals
            .structured
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
            mut goal_before,
            verified_at,
            turn_ledger_revision,
            verification_infrastructure_error,
        } = state;
        if !self.config.subagents.long_horizon {
            return false;
        }
        // Fold any block declared this turn into the baseline *before* anything
        // reads it. Every path below either keeps `goal_before` or restores it,
        // so applying the block here is what makes it durable — and it must be
        // durable, or the next turn re-activates a step the model already
        // reported as impossible and the drive loops on it forever.
        let mut blocked_this_turn = false;
        if let Some(prerequisite) = self.pending_block.take()
            && let Some(baseline) = goal_before.as_mut()
        {
            baseline.block_active(prerequisite);
            blocked_this_turn = true;
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
        // Phase C: same obligation as the interactive settle path — a done-claim
        // via update_plan or heuristic advance is not enough without a green seal
        // and a non-stalled turn. Skeptic (below) is an extra gate on top.
        let verified_clean = matches!(self.report.last_verify, Some(true));
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
        // Trivial-diff exemption: a full second-model review round-trip buys
        // nothing when the turn's net change is tiny and verify already passed
        // — the failures the gate catches (wrong artifact, stub stand-ins,
        // unhandled required cases) need more than a few bytes of diff to hide
        // in. Sum the byte deltas of this turn's changes; a delete or a
        // digest-only change (mode/mtime noise) counts as its full length.
        // Bounded by `SKEPTIC_TRIVIAL_DIFF_BYTES`; anything bigger reviews as
        // before. Statuses flips only (no net file change) can't reach here —
        // `clean_success` requires a verified change.
        let trivial_diff = {
            // Prose-only paths (docs, `.hi/memory.md`, skills) must not push a
            // one-line code fix over the trivial-diff exemption — coding-memory
            // and skill curation write those after verify by design.
            let changed_bytes: u64 = self
                .workspace
                .last_file_changes
                .iter()
                .filter(|change| !crate::verify::is_prose_only_path(&change.path))
                .map(|change| match (change.before_len, change.after_len) {
                    (Some(before), Some(after)) => before.abs_diff(after),
                    (_, Some(after)) => after,
                    (Some(before), None) => before,
                    (None, None) => 0,
                })
                .sum();
            changed_bytes <= crate::goal::SKEPTIC_TRIVIAL_DIFF_BYTES
        };
        if clean_success && trivial_diff {
            ui.status("🔍 skeptic skipped — trivial diff under verified pass");
        }
        if clean_success
            && !trivial_diff
            && let Some((objective, sub_goal, prior_notes)) = goal_before.as_ref().and_then(|g| {
                if !g.team || g.paused || g.status != GoalStatus::Active {
                    return None;
                }
                let sg = g.active_sub_goal()?;
                Some((
                    g.objective.clone(),
                    sg.description.clone(),
                    sg.notes.clone(),
                ))
            })
        {
            match self.skeptic_gate(&objective, &sub_goal, &prior_notes).await {
                SkepticVerdict::Object(items) => {
                    let objections = items.join("\n");
                    // Objection: revert the turn's goal progress and record it.
                    self.goals.structured = goal_before;
                    if let Some(goal) = self.goals.structured.as_mut() {
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
                    self.report.last_turn_telemetry.skeptic_last_status =
                        Some(SkepticStatus::Objected);
                    return false;
                }
                SkepticVerdict::Escalate(items) => {
                    // Unfixable-by-retry: skip the step with a visible scar
                    // and keep the run moving — retry-looping a contradiction
                    // wastes the budget, and parking to wait for the user is
                    // the worse failure for an unattended run. The escalation
                    // reasons land in the step's notes and the status line.
                    let reasons = items.join("\n");
                    self.goals.structured = goal_before;
                    if let Some(goal) = self.goals.structured.as_mut() {
                        goal.skeptic_escalations = goal.skeptic_escalations.saturating_add(1);
                        goal.last_skeptic_status = Some(SkepticStatus::Escalated);
                        goal.skip_active(format!(
                            "reviewer escalated — needs your judgment, skipped for now:\n{reasons}"
                        ));
                    }
                    let first = reasons.lines().next().unwrap_or("see notes");
                    ui.status(&format!(
                        "🛑 skeptic escalated — step needs your judgment, skipping past it: {first}"
                    ));
                    self.report.last_turn_telemetry.skeptic_last_status =
                        Some(SkepticStatus::Escalated);
                    self.refresh_system_message();
                    self.persist_goal(ui);
                    return false;
                }
                SkepticVerdict::Approve => {
                    if let Some(goal) = self.goals.structured.as_mut() {
                        goal.last_skeptic_status = Some(SkepticStatus::Approved);
                    }
                    self.report.last_turn_telemetry.skeptic_last_status =
                        Some(SkepticStatus::Approved);
                    ui.status("🔍 skeptic approved — advancing");
                }
                SkepticVerdict::Unavailable(reason) => {
                    if let Some(goal) = self.goals.structured.as_mut() {
                        goal.skeptic_unavailable = goal.skeptic_unavailable.saturating_add(1);
                        goal.last_skeptic_status = Some(SkepticStatus::Unavailable);
                    }
                    self.report.last_turn_telemetry.skeptic_unavailable_count = self
                        .report
                        .last_turn_telemetry
                        .skeptic_unavailable_count
                        .saturating_add(1);
                    self.report.last_turn_telemetry.skeptic_last_status =
                        Some(SkepticStatus::Unavailable);
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
            match self.runtime.reconcile_ledger_async().await {
                Ok(_) => {
                    let (revision, digest, changes) = {
                        let mut ledger = self.runtime.ledger();
                        (
                            ledger.revision(),
                            ledger.workspace_revision(),
                            ledger.changes_since(turn_ledger_revision),
                        )
                    };
                    let current_pass = verified_at.is_some_and(|(verified_revision, verified)| {
                        *verified_revision == revision && verified == &digest
                    });
                    self.workspace.last_changed_files =
                        changes.iter().map(|change| change.path.clone()).collect();
                    self.workspace.last_file_changes = changes;
                    if !current_pass {
                        self.report.last_verify = None;
                        clean_success = false;
                        verification_invalidated = true;
                        self.goals.structured = goal_before.clone();
                        self.refresh_system_message();
                        ui.status(
                            "workspace changed while completion review was running; goal progress was not advanced",
                        );
                    }
                }
                Err(error) => {
                    self.report.last_verify = None;
                    clean_success = false;
                    verification_invalidated = true;
                    self.goals.structured = goal_before.clone();
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
            if let Some(reviewed) = self.goals.structured.as_ref() {
                proposal.skeptic_objections = reviewed.skeptic_objections;
                proposal.skeptic_unavailable = reviewed.skeptic_unavailable;
                proposal.last_skeptic_status = reviewed.last_skeptic_status;
            }
            self.goals.structured = Some(proposal);
        }
        if !clean_success {
            // Defensive restoration for future mutations of the live goal
            // inside a turn. Today update_plan remains entirely provisional,
            // but the failure path stays explicitly anchored to the pre-turn
            // goal before any neutral/failure return.
            self.goals.structured = goal_before.clone();
        }
        // A clean read-only turn (investigation, Q&A — no edits, no verify,
        // no stall) is neutral: neither advance nor record failure. The sub-goal
        // stays active for the next turn, which should do the actual work.
        let no_edit_neutral = self.report.last_verify.is_none()
            && !stalled_unfinished
            && !stalled_repeating
            && !hit_step_cap
            && self.workspace.last_changed_files.is_empty();
        if no_edit_neutral {
            // Declaring a block is itself the turn's outcome, and a turn that
            // does only that changes no files — so it lands here. Persist it,
            // or the block survives in memory only and a restart re-activates a
            // step the model already reported as impossible.
            if blocked_this_turn {
                self.refresh_system_message();
                self.persist_goal(ui);
            }
            return verification_invalidated;
        }
        if clean_success {
            // Approve (or gate off): advance as today. If `update_plan` already
            // advanced the goal this turn, don't advance again (skips a sub-goal).
            if !plan_updated_goal && let Some(goal) = self.goals.structured.as_mut() {
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
                self.goals.structured.as_ref().map(|g| g.status),
                Some(GoalStatus::Done)
            ) {
                self.audit_goal_completion(ui).await;
            }
            if matches!(
                self.goals.structured.as_ref().map(|g| g.status),
                Some(GoalStatus::Done)
            ) {
                ui.status("✓ long-horizon goal complete");
            } else if plan_updated_goal
                && let (Some(before), Some(after)) = (
                    start_active_index,
                    self.goals.structured.as_ref().and_then(Goal::active_index),
                )
                && after > before
                && let Some(goal) = self.goals.structured.as_ref()
            {
                ui.status(&format!(
                    "✓ sub-goal {}/{} done — advancing",
                    before + 1,
                    goal.sub_goals.len().max(before + 1)
                ));
            }
            // A verified turn that left the same step active landed real work
            // without finishing the milestone. That is the only signal an
            // oversized step gives when the model ends its turns cleanly — it
            // never trips the step cap, so `cap_continuations` stays zero while
            // the step absorbs turn after turn. Enough of them means the step is
            // too large rather than merely slow, so split it.
            let still_on_same_step = self
                .goals
                .structured
                .as_ref()
                .and_then(Goal::active_index)
                .zip(start_active_index)
                .is_some_and(|(after, before)| after == before);
            if still_on_same_step && !self.workspace.last_changed_files.is_empty() {
                let oversized = self
                    .goals
                    .structured
                    .as_mut()
                    .is_some_and(Goal::record_productive_turn);
                let split_desc = self.goals.structured.as_ref().and_then(|g| {
                    let active = g.active_index()?;
                    let sg = &g.sub_goals[active];
                    (oversized
                        && sg.split_depth < crate::goal::MAX_SPLIT_DEPTH
                        && self.config.subagents.planner_model.is_some())
                    .then(|| sg.description.clone())
                });
                if let Some(desc) = split_desc
                    && let Ok(sub_steps) = self.decompose_milestone(&desc).await
                {
                    let spliced = self
                        .goals
                        .structured
                        .as_mut()
                        .map(|g| g.decompose_active(&sub_steps))
                        .unwrap_or(0);
                    if spliced >= 2 {
                        ui.status(&format!(
                            "🧩 milestone kept absorbing verified work without finishing — split into {spliced} turn-sized sub-steps"
                        ));
                    }
                }
            }
            self.refresh_system_message();
            self.persist_goal(ui);
            return verification_invalidated;
        }
        // A step-capped turn that made real progress (file changes) is a
        // continuation, not a failure: the milestone is bigger than one turn.
        // The work is on disk; the next drive turn resumes it. Only when a
        // sub-goal keeps capping out past MAX_CAP_CONTINUATIONS — or caps with
        // zero progress — does the retry/skip machinery judge it. Incrementing
        // the counter also changes goal state, which resets the frontend
        // drive-stall counter so a long milestone isn't parked mid-build.
        if hit_step_cap {
            let made_progress = !self.workspace.last_changed_files.is_empty();
            // A milestone that keeps hitting the step cap *while making progress*
            // is too big for one turn: decompose it into turn-sized sub-steps
            // rather than grind it out over dozens of turns. Snapshot the decision
            // inputs first — the planner call below borrows `self`, so it can't
            // overlap the goal borrow.
            let split_desc = self.goals.structured.as_ref().and_then(|g| {
                let active = g.active_index()?;
                let sg = &g.sub_goals[active];
                (made_progress
                    && sg.cap_continuations + 1 >= crate::goal::DECOMPOSE_AFTER_CONTINUATIONS
                    && sg.split_depth < crate::goal::MAX_SPLIT_DEPTH
                    && self.config.subagents.planner_model.is_some())
                .then(|| sg.description.clone())
            });
            if let Some(desc) = split_desc
                && let Ok(sub_steps) = self.decompose_milestone(&desc).await
            {
                let spliced = self
                    .goals
                    .structured
                    .as_mut()
                    .map(|g| g.decompose_active(&sub_steps))
                    .unwrap_or(0);
                if spliced >= 2 {
                    ui.status(&format!(
                        "🧩 milestone too large for one turn — split into {spliced} turn-sized sub-steps"
                    ));
                    self.refresh_system_message();
                    self.persist_goal(ui);
                    return verification_invalidated;
                }
            }
            // Otherwise treat the capped turn as a continuation. A turn that
            // landed edits is real progress and resets the barren counter; a
            // capped turn with no net file change is "barren". Hitting the step
            // cap means "more work to do," not "failed", so we continue across
            // turns as long as the milestone keeps making progress — only a run
            // of barren caps (the model can't land edits) or the generous safety
            // ceiling ends the continuation and hands it to the retry/skip machinery.
            if let Some(goal) = self.goals.structured.as_mut()
                && let Some(active) = goal.active_index()
            {
                let sub_goal = &mut goal.sub_goals[active];
                if made_progress {
                    sub_goal.barren_caps = 0;
                } else {
                    sub_goal.barren_caps = sub_goal.barren_caps.saturating_add(1);
                    // Steer the next turn to implement rather than keep exploring.
                    crate::goal::push_note_deduped(sub_goal, crate::goal::BARREN_CAP_NOTE);
                }
                if sub_goal.barren_caps < crate::goal::MAX_BARREN_CAPS
                    && sub_goal.cap_continuations < crate::goal::MAX_CAP_CONTINUATIONS
                {
                    sub_goal.cap_continuations = sub_goal.cap_continuations.saturating_add(1);
                    let n = sub_goal.cap_continuations;
                    let msg = if made_progress {
                        format!(
                            "⏳ milestone spans turns: hit the step cap with progress — continuing ({n}/{})",
                            crate::goal::MAX_CAP_CONTINUATIONS
                        )
                    } else {
                        format!(
                            "⏳ milestone spans turns: hit the step cap while exploring ({}/{} barren) — continuing; land concrete edits next turn",
                            sub_goal.barren_caps,
                            crate::goal::MAX_BARREN_CAPS
                        )
                    };
                    ui.status(&msg);
                    self.refresh_system_message();
                    self.persist_goal(ui);
                    return verification_invalidated;
                }
            }
        }
        // A stalled or cap-hit turn, or a verify failure that ended the turn,
        // records a sub-goal attempt so the next turn sees the prior note. If
        // the budget is exhausted, the sub-goal (and goal) is marked Failed.
        // Verification never reached a verdict, so there is nothing to hold
        // against the work. Charging this to the retry budget is what marked a
        // healthy checklist `Failed` step by step when the only real defect was
        // a verify command that could not finish. Record it, keep the budget
        // intact, and park the drive once the checks have failed to conclude
        // often enough that they — not the model — are clearly the problem.
        if verification_infrastructure_error {
            let note = "verification could not run to a verdict, so this turn's work was never judged — the checks themselves need attention (see the status line)";
            let may_continue = match self.goals.structured.as_mut() {
                Some(goal) => goal.record_unjudged(note),
                None => return verification_invalidated,
            };
            if may_continue {
                ui.status(
                    "⚠ verification never reached a verdict — retrying without charging the sub-goal",
                );
            } else if let Some(goal) = self.goals.structured.as_mut() {
                goal.pause(crate::goal::GoalPauseReason::Infra);
                ui.status(&format!(
                    "⏸ verification has failed to reach a verdict {} turns running — pausing the goal rather than burning turns on checks that never conclude. Fix the verify command (or raise its timeout), then /goal resume.",
                    crate::goal::MAX_UNJUDGED_TURNS
                ));
            }
            self.refresh_system_message();
            self.persist_goal(ui);
            return verification_invalidated;
        }
        let reason = if hit_step_cap {
            "hit the per-turn step cap"
        } else if self.report.last_verify == Some(false) {
            "verification failed and the turn ended without fixing it"
        } else if stalled_repeating {
            "stalled repeating the same tool call"
        } else if stalled_unfinished {
            "ended without completing the requested work"
        } else if self.report.last_verify.is_none() && !self.workspace.last_changed_files.is_empty()
        {
            "ended with unverified workspace changes"
        } else {
            "verification failed and the turn ended without fixing it"
        };
        let can_retry = match self.goals.structured.as_mut() {
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
                .goals
                .structured
                .as_mut()
                .is_some_and(Goal::continue_past_failure);
            // Skipping keeps a long run alive when one milestone is stuck — but
            // a run that has never completed *anything* is thrashing, and its
            // advancing cursor is indistinguishable from progress in every
            // surface the user sees. Park instead of walking the whole plan.
            let thrashing = self
                .goals
                .structured
                .as_ref()
                .is_some_and(Goal::is_thrashing);
            if thrashing {
                if let Some(goal) = self.goals.structured.as_mut() {
                    let skips = goal.consecutive_skips;
                    goal.pause(crate::goal::GoalPauseReason::Stall);
                    ui.status(&format!(
                        "⏸ {skips} sub-goals abandoned with none completed — pausing rather than walking the rest of the plan. The last failure was: {reason}. /goal status to review, then /goal resume."
                    ));
                }
            } else if skipped {
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

    /// Handle a `block_step` tool call: set the active sub-goal aside as
    /// [`GoalStatus::Blocked`] with the named prerequisite.
    ///
    /// This exists so a missing dependency stops costing retries. Without it
    /// the model's only options are to keep failing (three attempts, then the
    /// step is marked `Failed` — reporting the *work* as rejected when the real
    /// problem was an absent database) or to write a stub that skips the
    /// required check, which is worse because it looks like success.
    pub(crate) fn handle_block_step(&mut self, arguments: &str) -> hi_tools::ToolOutcome {
        #[derive(serde::Deserialize)]
        struct BlockArgs {
            prerequisite: String,
        }
        let prerequisite = match serde_json::from_str::<BlockArgs>(arguments) {
            Ok(args) => args.prerequisite.trim().to_string(),
            Err(err) => {
                return decision_tool_outcome(
                    format!("Error: bad block_step arguments: {err}"),
                    hi_tools::ToolStatus::Failed,
                );
            }
        };
        if prerequisite.is_empty() {
            return decision_tool_outcome(
                "Error: block_step needs a non-empty prerequisite".to_string(),
                hi_tools::ToolStatus::Failed,
            );
        }
        if !self.config.subagents.long_horizon {
            return decision_tool_outcome(
                "Error: block_step only applies to a long-horizon goal; none is active".to_string(),
                hi_tools::ToolStatus::Failed,
            );
        }
        let Some(goal) = self.goals.structured.as_mut() else {
            return decision_tool_outcome(
                "Error: no long-horizon goal is set, so there is no step to block".to_string(),
                hi_tools::ToolStatus::Failed,
            );
        };
        if goal.active_index().is_none() {
            return decision_tool_outcome(
                "Error: no sub-goal is active, so there is nothing to block".to_string(),
                hi_tools::ToolStatus::Failed,
            );
        }
        let more_work = goal.block_active(&prerequisite);
        let blocked = goal.blocked_steps().len();
        // Survive the turn-end rollback (see `Agent::pending_block`).
        self.pending_block = Some(prerequisite.clone());
        self.refresh_system_message();
        let message = if more_work {
            format!(
                "Step set aside as blocked on: {prerequisite}. Moving to the next step ({blocked} blocked so far). Do not retry this one or stub it out."
            )
        } else {
            format!(
                "Step set aside as blocked on: {prerequisite}. No drivable steps remain ({blocked} blocked) — the goal stops here until the prerequisites are satisfied."
            )
        };
        decision_tool_outcome(message, hi_tools::ToolStatus::Succeeded)
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
