//! Transient and structured goal state, persistence, and status.

use anyhow::Result;

use crate::{decision::DecisionLog, goal::Goal};

impl crate::Agent {
    /// Current transient session goal, if any.
    pub fn goal(&self) -> Option<&str> {
        self.goals.free_text.as_deref()
    }

    /// The durable intra-session decision log (recorded via `record_decision`),
    /// injected into the system prompt each turn and preserved across compaction.
    pub fn decisions(&self) -> &DecisionLog {
        &self.decisions
    }

    /// Set or clear the transient session goal and inject it into the system prompt.
    pub fn set_goal(&mut self, goal: Option<String>) {
        self.goals.set_free_text(goal);
        self.refresh_system_message();
    }

    /// Set or clear the transient session goal, first clearing any persisted
    /// structured long-horizon goal so it cannot reappear on a later resume.
    pub fn set_transient_goal(&mut self, goal: Option<String>) -> Result<()> {
        self.set_structured_goal(None)?;
        self.set_goal(goal);
        Ok(())
    }

    /// Set or clear a structured long-horizon goal (decomposed into sub-goals).
    /// Only takes effect when `config.subagents.long_horizon` is on; when set, the goal's
    /// state is injected into the system prompt each turn so the agent resumes
    /// the active sub-goal. Returns whether it was accepted.
    pub fn set_structured_goal(&mut self, mut goal: Option<Goal>) -> Result<bool> {
        if !self.config.subagents.long_horizon && goal.is_some() {
            return Ok(false);
        }
        let migrated_legacy_goal_budget = goal
            .as_mut()
            .is_some_and(Goal::clear_legacy_automatic_budget);
        if let Some(session) = self.session.as_mut() {
            if let Some(g) = &goal {
                session.record_goal(g)?;
            } else {
                session.clear_goal()?;
            }
        }
        // A structured goal is the durable replacement for the transient
        // `/goal <text>` prompt injection. Keeping both would make every turn
        // carry two competing objectives after the user switches modes.
        // Clear the transient value only after persistence succeeds so a failed
        // session write leaves the in-memory state unchanged.
        if goal.is_some() {
            self.goals.free_text = None;
        }
        self.goals.set_structured(goal);
        self.pending_legacy_goal_budget_migration =
            migrated_legacy_goal_budget && self.session.is_none();
        self.refresh_system_message();
        Ok(true)
    }

    /// The structured long-horizon goal, if any (for persistence/observability).
    pub fn structured_goal(&self) -> Option<&Goal> {
        self.goals.structured.as_ref()
    }

    /// Pause or resume the structured goal without losing progress: a paused goal
    /// is dropped from the system prompt and the driver leaves it alone, but its
    /// sub-goal progress is retained and persisted so `/goal resume` picks up
    /// exactly where it left off. Returns whether there was a goal to update.
    pub fn set_goal_paused(&mut self, paused: bool) -> bool {
        self.try_set_goal_pause_reason(if paused {
            crate::GoalPauseReason::User
        } else {
            crate::GoalPauseReason::None
        })
        .unwrap_or(false)
    }

    /// Pause/resume with a typed reason (`User`, `Stall`, `Review`, …).
    pub fn set_goal_pause_reason(&mut self, reason: crate::GoalPauseReason) -> bool {
        self.try_set_goal_pause_reason(reason).unwrap_or(false)
    }

    /// Fallible form of [`Self::set_goal_pause_reason`] for command frontends.
    /// The in-memory mutation is rolled back when the durable goal record cannot
    /// be written, so a successful-looking pause can never disappear on resume.
    pub fn try_set_goal_pause_reason(&mut self, reason: crate::GoalPauseReason) -> Result<bool> {
        self.update_structured_goal(|goal| {
            if matches!(reason, crate::GoalPauseReason::None) {
                goal.resume();
            } else {
                goal.pause(reason);
            }
        })
    }

    /// Mutate the structured goal and persist (events, edits, etc.).
    pub fn update_structured_goal(&mut self, f: impl FnOnce(&mut Goal)) -> Result<bool> {
        let Some(previous) = self.goals.structured.clone() else {
            return Ok(false);
        };
        let snapshot = {
            let goal = self
                .goals
                .structured
                .as_mut()
                .expect("structured goal existed immediately before mutation");
            f(goal);
            goal.clone()
        };
        // Avoid rewriting the session record and rebuilding the prompt when a
        // command repeats the current setting (`/goal team on`, for example).
        if snapshot == previous {
            return Ok(true);
        }
        if let Some(session) = self.session.as_mut()
            && let Err(err) = session.record_goal(&snapshot)
        {
            self.goals.structured = Some(previous);
            self.refresh_system_message();
            return Err(err);
        }
        self.refresh_system_message();
        Ok(true)
    }

    /// Export goal checklist markdown to `.hi/goal-plan.md`.
    pub fn export_goal_plan(&mut self) -> Result<Option<std::path::PathBuf>> {
        let Some(goal) = self.goals.structured.as_ref() else {
            return Ok(None);
        };
        let path = goal.export_markdown_to(self.workspace_root())?;
        let _ = self.update_structured_goal(|g| {
            g.push_event("export", format!("wrote {}", path.display()));
        });
        Ok(Some(path))
    }

    /// Turn the `/goal team` skeptic gate on/off for the active goal. Persists with
    /// the goal (so a resumed goal remembers it) and refreshes the system message.
    /// Returns `false` if there's no active goal.
    pub fn set_goal_team(&mut self, on: bool) -> bool {
        self.try_set_goal_team(on).unwrap_or(false)
    }

    /// Fallible form of [`Self::set_goal_team`] for command frontends.
    pub fn try_set_goal_team(&mut self, on: bool) -> Result<bool> {
        self.update_structured_goal(|goal| goal.team = on)
    }

    /// Elevate Goal-drive turns to Always (YOLO) for this goal, restoring the
    /// previous permission mode at each turn end.
    pub fn try_set_goal_unattended(&mut self, on: bool) -> Result<bool> {
        self.update_structured_goal(|goal| goal.unattended = on)
    }

    /// Set (or clear, with `None`) a ceiling on how many sub-goals the goal's plan
    /// may grow to. `None` is the default — no limit, the plan keeps expanding as the
    /// agent discovers work. Persisted with the goal. Returns whether there was a
    /// goal to update.
    /// Set (or clear) the goal's drive-turn budget.
    ///
    /// Setting a budget also resets the spend and clears a budget pause, so
    /// `/goal budget 20` after a park means "twenty more turns" rather than
    /// re-parking immediately on the already-spent count.
    pub fn set_goal_turn_budget(&mut self, budget: Option<u32>) -> bool {
        self.try_set_goal_turn_budget(budget).unwrap_or(false)
    }

    /// Fallible form of [`Self::set_goal_turn_budget`] for command frontends.
    pub fn try_set_goal_turn_budget(&mut self, budget: Option<u32>) -> Result<bool> {
        self.update_structured_goal(|goal| {
            goal.turn_budget = budget;
            // Preserve the legacy marker only for session migration; every
            // explicit command installs exactly the user's value.
            goal.budget_auto = false;
            goal.turns_spent = 0;
            if goal.pause_reason == crate::goal::GoalPauseReason::Budget {
                goal.resume();
            }
        })
    }

    pub fn set_goal_step_limit(&mut self, limit: Option<usize>) -> bool {
        self.try_set_goal_step_limit(limit).unwrap_or(false)
    }

    /// Fallible form of [`Self::set_goal_step_limit`] for command frontends.
    pub fn try_set_goal_step_limit(&mut self, limit: Option<usize>) -> Result<bool> {
        self.update_structured_goal(|goal| goal.step_limit = limit)
    }

    /// The per-session turn limit (`/turns`). `None` = unlimited.
    pub fn max_turns(&self) -> Option<u32> {
        self.config.max_turns
    }

    /// How many turns have completed in this session so far.
    pub fn turn_count(&self) -> u32 {
        self.turn_count
    }

    /// Set (or clear, with `None`) the per-session turn limit. Live only — not
    /// persisted with the goal. Takes effect on the next `run_turn` entry.
    pub fn set_max_turns(&mut self, limit: Option<u32>) {
        self.config.max_turns = limit;
    }

    /// One-line goal summary for status surfaces: the structured goal's
    /// progress ("objective — 2/7 sub-goals done", with a paused marker) when one
    /// is set, else the transient goal string, else "off".
    pub fn goal_summary(&self) -> String {
        if let Some(g) = &self.goals.structured {
            let done = g
                .sub_goals
                .iter()
                .filter(|s| s.status == crate::GoalStatus::Done)
                .count();
            let paused = if g.is_paused() {
                format!(" · paused({})", g.pause_reason.as_str())
            } else {
                String::new()
            };
            let skeptic = if g.team {
                format!(
                    " · skeptic: {} unavailable, last {}",
                    g.skeptic_unavailable,
                    g.last_skeptic_status
                        .map(|status| format!("{status:?}"))
                        .unwrap_or_else(|| "not run".into())
                )
            } else {
                String::new()
            };
            let complete = if g.objective_complete {
                " · objective✓"
            } else {
                ""
            };
            return format!(
                "{} — {}/{} sub-goals done{paused}{skeptic}{complete}",
                g.objective,
                done,
                g.sub_goals.len()
            );
        }
        self.goals
            .free_text
            .clone()
            .unwrap_or_else(|| "off".to_string())
    }
}
