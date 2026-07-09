//! Goal-level turn-end hooks: the long-horizon driver (`goal_turn_end`)
//! that advances/retries the active sub-goal, and `handle_record_decision`
//! for the `record_decision` tool.

use crate::Ui;
use crate::decision::Decision;
use crate::goal::{DEFAULT_SUBGOAL_RETRIES, Goal, GoalStatus};

impl crate::Agent {
    /// Long-horizon driver — called at turn end. When a structured goal is set
    /// and `long_horizon` is on, advance or retry the active sub-goal based on
    /// the turn's outcome, so the next turn resumes at the right sub-goal (with
    /// prior-attempt notes if it stalled, so the model doesn't repeat a failed
    /// approach). The verify retry itself happens *within* the turn (the 'turn
    /// loop re-runs the model on a verify failure); this hook handles the
    /// goal-level progression once the turn settles.
    pub(crate) async fn goal_turn_end(
        &mut self,
        stalled_unfinished: bool,
        stalled_repeating: bool,
        hit_step_cap: bool,
        plan_updated_goal: bool,
        goal_before: Option<Goal>,
        ui: &mut dyn Ui,
    ) {
        if !self.config.long_horizon {
            return;
        }
        let max_retries = DEFAULT_SUBGOAL_RETRIES;
        // A turn that verified clean (or had no verify but made edits without
        // stalling) completes the active sub-goal → advance.
        let verified_clean = matches!(self.last_verify, Some(true));
        let no_verify_clean = self.last_verify.is_none()
            && !stalled_unfinished
            && !stalled_repeating
            && !hit_step_cap
            && !self.last_changed_files.is_empty();
        let clean_success = verified_clean || no_verify_clean;

        // Skeptic gate: on a clean-success turn, a second model reviews the work
        // before its progress stands. It reviews the sub-goal that was active AT
        // TURN START — because `update_plan` may have marked that sub-goal (or the
        // whole goal) done mid-turn, and the model's own "done" claim is exactly
        // what a skeptic should second-guess. On an objection we revert the turn's
        // goal progress (restore the pre-turn goal) and record the objections as a
        // retry note; the edits stay on disk for the next turn to build on.
        // Fail-open — any reviewer error/timeout/unparseable reply approves.
        if clean_success
            && self.has_skeptic()
            && let Some((objective, sub_goal)) = goal_before.as_ref().and_then(|g| {
                if !g.team || g.paused || g.status != GoalStatus::Active {
                    return None;
                }
                let sg = g.active_sub_goal()?;
                Some((g.objective.clone(), sg.description.clone()))
            })
        {
            match self.skeptic_gate(&objective, &sub_goal).await {
                Some(objections) => {
                    // Objection: revert the turn's goal progress and record it.
                    self.structured_goal = goal_before;
                    if let Some(goal) = self.structured_goal.as_mut() {
                        goal.skeptic_objections = goal.skeptic_objections.saturating_add(1);
                        goal.record_failure(
                            format!("reviewer objected — address then continue:\n{objections}"),
                            max_retries,
                        );
                    }
                    let first = objections.lines().next().unwrap_or("see notes");
                    ui.status(&format!("🔍 skeptic objected — retrying: {first}"));
                    self.refresh_system_message();
                    self.persist_goal(ui);
                    return;
                }
                // Approved (or a fail-open error): note it and let the advance
                // stand — fall through to the normal advance/retry logic.
                None => ui.status("🔍 skeptic reviewed — advancing"),
            }
        }

        // Normal advance/retry bookkeeping, on the CURRENT goal.
        {
            let Some(goal) = self.structured_goal.as_ref() else {
                return;
            };
            if goal.paused {
                return; // Paused by the user — hold progress.
            }
            if goal.status != GoalStatus::Active {
                return; // Done/failed (perhaps via update_plan) — nothing to drive.
            }
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
            return;
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
            if matches!(
                self.structured_goal.as_ref().map(|g| g.status),
                Some(GoalStatus::Done)
            ) {
                ui.status("✓ long-horizon goal complete");
            }
            self.refresh_system_message();
            self.persist_goal(ui);
            return;
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
        } else {
            "verification failed and the turn ended without fixing it"
        };
        let can_retry = match self.structured_goal.as_mut() {
            Some(goal) => goal.record_failure(reason, max_retries),
            None => return,
        };
        if can_retry {
            ui.status(&format!(
                "↻ sub-goal failed this turn ({reason}) — will retry next turn, don't repeat the same approach"
            ));
        } else {
            ui.status(&format!(
                "✗ sub-goal exhausted its retry budget ({reason}) — marked failed; /goal to revise or continue past it"
            ));
        }
        self.refresh_system_message();
        self.persist_goal(ui);
    }

    /// Handle a `record_decision` tool call: parse the args, append to the
    /// durable decision log (which feeds the system prompt), and return a
    /// terse confirmation for the model. Malformed args yield an error string
    /// (the model sees it and can retry), not a panic.
    pub(crate) fn handle_record_decision(&mut self, arguments: &str) -> String {
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
                    return "Error: record_decision needs a non-empty summary".to_string();
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
                    return format!("Error: couldn't persist decision: {err}");
                }
                self.decisions = next;
                // Refresh the system prompt so the decision is injected on the
                // next turn (and visible to the model immediately in history).
                self.refresh_system_message();
                "Decision recorded — it will persist across compaction.".to_string()
            }
            Err(err) => format!("Error: bad record_decision arguments: {err}"),
        }
    }
}
