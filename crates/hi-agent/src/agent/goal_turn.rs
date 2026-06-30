//! Goal-level turn-end hooks: the long-horizon driver (`goal_turn_end`)
//! that advances/retries the active sub-goal, and `handle_record_decision`
//! for the `record_decision` tool.

use crate::decision::Decision;
use crate::goal::{DEFAULT_SUBGOAL_RETRIES, GoalStatus};
use crate::Ui;

impl crate::Agent {
    /// Long-horizon driver — called at turn end. When a structured goal is set
    /// and `long_horizon` is on, advance or retry the active sub-goal based on
    /// the turn's outcome, so the next turn resumes at the right sub-goal (with
    /// prior-attempt notes if it stalled, so the model doesn't repeat a failed
    /// approach). The verify retry itself happens *within* the turn (the 'turn
    /// loop re-runs the model on a verify failure); this hook handles the
    /// goal-level progression once the turn settles.
    pub(crate) fn goal_turn_end(
        &mut self,
        _stalled_unfinished: bool,
        stalled_repeating: bool,
        hit_step_cap: bool,
        plan_updated_goal: bool,
        ui: &mut dyn Ui,
    ) {
        if !self.config.long_horizon {
            return;
        }
        let Some(goal) = self.structured_goal.as_mut() else {
            return;
        };
        if goal.status != GoalStatus::Active {
            return; // Already done or failed — nothing to drive.
        }
        let max_retries = DEFAULT_SUBGOAL_RETRIES;
        // A turn that verified clean (or had no verify but made edits without
        // stalling) completes the active sub-goal → advance.
        let verified_clean = matches!(self.last_verify, Some(true));
        let no_verify_clean = self.last_verify.is_none()
            && !stalled_repeating
            && !hit_step_cap
            && !self.last_changed_files.is_empty();
        // A clean read-only turn (investigation, Q&A — no edits, no verify,
        // no stall) is neutral: neither advance nor record failure. The sub-goal
        // stays active for the next turn, which should do the actual work.
        let no_edit_neutral = self.last_verify.is_none()
            && !stalled_repeating
            && !hit_step_cap
            && self.last_changed_files.is_empty();
        if no_edit_neutral {
            return;
        }
        if verified_clean || no_verify_clean {
            // If the model's update_plan already advanced the goal during this
            // turn (apply_plan_to_goal marked the active sub-goal done and
            // activated the next), don't advance again — that would skip the
            // newly-activated sub-goal. Otherwise, advance normally.
            if !plan_updated_goal {
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
            if goal.status == GoalStatus::Done {
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
        } else if stalled_repeating {
            "stalled repeating the same tool call"
        } else {
            "verification failed and the turn ended without fixing it"
        };
        let can_retry = goal.record_failure(reason, max_retries);
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
                self.decisions.record(Decision {
                    summary,
                    rationale: args.rationale.trim().to_string(),
                    files: args.files,
                });
                // Refresh the system prompt so the decision is injected on the
                // next turn (and visible to the model immediately in history).
                self.refresh_system_message();
                "Decision recorded — it will persist across compaction.".to_string()
            }
            Err(err) => format!("Error: bad record_decision arguments: {err}"),
        }
    }

}
