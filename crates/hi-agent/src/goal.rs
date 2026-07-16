//! Long-horizon agency: a structured goal the agent decomposes into sub-goals,
//! drives across turns, retries on failure, and persists across sessions.
//!
//! This is the foundation for multi-turn, multi-hour tasks ("refactor this
//! crate", "land this feature across these files"). The single-turn loop
//! (`run_turn`) works one user turn at a time; a `Goal` gives it a persistent
//! objective it resumes coherently: the active sub-goal is injected into the
//! system prompt each turn, the model works it, and the plan updates map back
//! to sub-goal status so the agent advances (or retries) across turns.
//!
//! Feature-gated behind `AgentConfig::long_horizon` (default off) so the
//! existing single-turn behavior is unchanged while this stabilizes.
//!
//! The state machine is unit-tested here; the deep `run_turn` outer-loop
//! integration (driving sub-goals across turns, retry-nudging on failure) is
//! the next step and lives in the agent loop. This module provides the typed
//! state and the rules.

use serde::{Deserialize, Serialize};

/// The status of a sub-goal (and the overall goal).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    /// Not started.
    Pending,
    /// Currently being worked.
    Active,
    /// Completed successfully.
    Done,
    /// Exhausted its retry budget without succeeding; skipped or needs the user.
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkepticStatus {
    Approved,
    Objected,
    Unavailable,
}

/// One step of a decomposed goal. The agent works sub-goals in order; a failed
/// sub-goal is retried up to `attempts` before being marked `Failed`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubGoal {
    /// A short description of what this step accomplishes.
    pub description: String,
    pub status: GoalStatus,
    /// How many times this sub-goal has been attempted (incremented on each
    /// retry; reset isn't sensible — the count records history).
    #[serde(default)]
    pub attempts: u32,
    /// Free-form notes from the agent (e.g. why a retry was needed, or a dead
    /// end hit). Carried into the system prompt so the model doesn't repeat a
    /// failed approach.
    #[serde(default)]
    pub notes: Vec<String>,
    /// How many turns hit the per-turn step cap *while making real progress*
    /// (file changes) on this sub-goal. Such turns are continuations — the
    /// milestone is bigger than one turn — not failures, so they don't burn
    /// the retry budget until [`MAX_CAP_CONTINUATIONS`]. Incrementing also
    /// marks the goal as changed, which keeps the frontend drive-stall
    /// counter from parking a long multi-turn milestone. `#[serde(default)]`.
    #[serde(default)]
    pub cap_continuations: u32,
}

/// A structured, multi-step objective that persists across turns and sessions.
/// Distinct from the transient `Agent.goal` string (which is just a prompt
/// injection): a `Goal` is decomposed, tracked, and resumed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Goal {
    /// The high-level objective, as the user stated it.
    pub objective: String,
    /// The ordered sub-goals the agent decomposed the objective into.
    pub sub_goals: Vec<SubGoal>,
    /// The overall status — `Done` when all sub-goals are done, `Failed` if a
    /// sub-goal failed and the agent gave up, otherwise `Active`.
    pub status: GoalStatus,
    /// Whether the goal is paused: progress is retained and persisted, but the
    /// goal stops steering turns (dropped from the system prompt) and the driver
    /// leaves it alone until resumed. Orthogonal to `status` — a re-derivation of
    /// status (e.g. `apply_plan_statuses`) never touches it. `/goal resume` clears
    /// it. `#[serde(default)]` so goals saved before pause/resume load as active.
    #[serde(default)]
    pub paused: bool,
    /// Optional user-set ceiling on how many sub-goals the plan may grow to (via
    /// `/goal limit <n>`). `None` (the default) means **no limit** — the plan keeps
    /// expanding as the agent discovers work, which is the point for long,
    /// adventurous objectives ("port this service to Rust"). Part of the contract,
    /// so it persists with the goal. `#[serde(default)]` for older saved goals.
    #[serde(default)]
    pub step_limit: Option<usize>,
    /// Whether the `/goal team` skeptic gate is active for this goal: a second
    /// model reviews each turn before it advances a sub-goal, and can send the work
    /// back to retry. **On by default for new goals** (`Goal::new`) — the gate
    /// exists precisely to second-guess the model's own "done" claims, and it works
    /// unconfigured (skeptic falls back to the session model). Toggled by
    /// `/goal team on|off` / `HI_GOAL_TEAM`. `#[serde(default)]` stays `false` so
    /// goals saved before the gate existed load exactly as they ran.
    #[serde(default)]
    pub team: bool,
    /// How many times the skeptic gate has sent the active work back to retry —
    /// observability for whether the gate is actually catching things.
    /// `#[serde(default)]`.
    #[serde(default)]
    pub skeptic_objections: u32,
    /// Reviewer failures are visible but do not block goal advancement.
    #[serde(default)]
    pub skeptic_unavailable: u32,
    #[serde(default)]
    pub last_skeptic_status: Option<SkepticStatus>,
    /// How many completion-audit rounds have appended missing work to this goal.
    /// Bounds the audit loop so a goal can't oscillate at the finish line forever.
    /// `#[serde(default)]` so older saved goals load at 0.
    #[serde(default)]
    pub audit_rounds: u32,
}

/// Default per-sub-goal retry budget: how many times to retry a failing sub-goal
/// (with a "reconsider, don't repeat" nudge) before marking it `Failed`.
pub const DEFAULT_SUBGOAL_RETRIES: u32 = 2;

/// The synthetic prompt frontends feed the agent between turns to keep an active
/// goal moving without the user re-prompting each step (the goal's checklist and
/// notes ride in the system prompt, so this stays short). Frontends compare the
/// input line against this constant to know a turn is auto-drive, not the user.
pub const GOAL_CONTINUE_PROMPT: &str = "Continue the long-horizon goal: complete the active \
sub-goal now, then call update_plan with the full goal checklist in its existing order — update \
statuses and append any newly discovered implementation steps.";

/// Stop auto-driving after this many consecutive drive turns that left the goal
/// state untouched (no advance, no retry note, no plan growth). The goal stays
/// active — the user's next message resumes the drive — but we don't burn turns
/// spinning in place. Sized generously: a hard step can legitimately spend a
/// few pure-investigation turns before its first edit, and stopping to wait
/// for a human is the worse failure mode for an unattended run.
pub const GOAL_DRIVE_STALL_LIMIT: u32 = 4;

/// Note recorded on a sub-goal the model claimed complete before it was ever
/// driven. Applied instead of the status flip; rendered by [`Goal::prompt_section`]
/// when the step becomes active so the drive turn starts skeptical.
pub const CLAIM_NOTE: &str = "the model previously claimed this step was already complete \
without it ever being driven — verify that claim against the actual work rather than trusting it";

/// Note recorded when a plan update tried to revert a completed sub-goal.
/// The revert is ignored; rework needs an explicit /goal revision.
pub const REGRESSION_NOTE: &str = "a later plan update tried to revert this completed step — \
ignored; reopen explicitly via /goal if rework is needed";

/// How many step-capped-but-productive turns a single sub-goal may take before
/// capped turns start burning its retry budget again. A big milestone (a whole
/// crate from scratch) legitimately spans several capped turns; a milestone
/// still capping out after this many is thrashing, and the normal
/// retry/skip machinery should judge it.
pub const MAX_CAP_CONTINUATIONS: u32 = 8;

impl Goal {
    /// Create a fresh goal with sub-goals all `Pending` except the first
    /// `Active`. The agent decomposes the objective into `sub_goal_descriptions`
    /// (one model call, done by the agent loop); this constructor takes the
    /// already-decomposed list.
    pub fn new(objective: impl Into<String>, sub_goal_descriptions: Vec<String>) -> Self {
        let sub_goals = sub_goal_descriptions
            .into_iter()
            .enumerate()
            .map(|(i, d)| SubGoal {
                description: d,
                status: if i == 0 {
                    GoalStatus::Active
                } else {
                    GoalStatus::Pending
                },
                attempts: 0,
                notes: Vec::new(),
                cap_continuations: 0,
            })
            .collect();
        Self {
            objective: objective.into(),
            sub_goals,
            status: GoalStatus::Active,
            paused: false,
            step_limit: None,
            team: true,
            skeptic_objections: 0,
            skeptic_unavailable: 0,
            last_skeptic_status: None,
            audit_rounds: 0,
        }
    }

    /// Whether frontends should keep auto-driving this goal between turns: it's
    /// still in progress, not paused by the user, and actually has steps. `Done`,
    /// `Failed`, paused, or empty goals are left alone.
    pub fn should_auto_drive(&self) -> bool {
        self.status == GoalStatus::Active && !self.paused && !self.sub_goals.is_empty()
    }

    /// The currently-active sub-goal, if any (the first `Active` one).
    pub fn active_sub_goal(&self) -> Option<&SubGoal> {
        self.sub_goals
            .iter()
            .find(|s| s.status == GoalStatus::Active)
    }

    /// The currently-active sub-goal index, if any.
    pub fn active_index(&self) -> Option<usize> {
        self.sub_goals
            .iter()
            .position(|s| s.status == GoalStatus::Active)
    }

    /// Mark the active sub-goal done and advance to the next (which becomes
    /// `Active`). If that was the last, the overall goal is `Done`.
    pub fn advance(&mut self) {
        if let Some(i) = self.active_index() {
            self.sub_goals[i].status = GoalStatus::Done;
            if let Some(next) = self.sub_goals.get_mut(i + 1) {
                next.status = GoalStatus::Active;
            } else {
                self.status = GoalStatus::Done;
            }
        }
    }

    /// Record an attempt on the active sub-goal (a verify failure the model
    /// couldn't fix, or a dead end). Returns `true` if the sub-goal still has
    /// retry budget (the agent should nudge "reconsider, don't repeat" and
    /// retry), `false` if it's now `Failed` (budget exhausted) — in which case
    /// the overall goal is also `Failed` unless the agent chooses to skip.
    pub fn record_failure(&mut self, note: impl Into<String>, max_retries: u32) -> bool {
        let Some(i) = self.active_index() else {
            return false;
        };
        let sg = &mut self.sub_goals[i];
        sg.attempts += 1;
        sg.notes.push(note.into());
        if sg.attempts > max_retries {
            sg.status = GoalStatus::Failed;
            self.status = GoalStatus::Failed;
            false
        } else {
            true
        }
    }

    /// Skip the active sub-goal (mark `Failed` with a note) and advance to the
    /// next, so a blocked step doesn't halt the whole goal. The overall goal
    /// stays `Active` unless this was the last sub-goal.
    pub fn skip_active(&mut self, note: impl Into<String>) {
        if let Some(i) = self.active_index() {
            self.sub_goals[i].status = GoalStatus::Failed;
            self.sub_goals[i].notes.push(note.into());
            if let Some(next) = self.sub_goals.get_mut(i + 1) {
                next.status = GoalStatus::Active;
            } else {
                self.status = GoalStatus::Failed;
            }
        }
    }

    /// Apply the model's `update_plan` statuses to the sub-goals. The model
    /// resubmits the whole list each time; this maps `done`/`active`/`pending`
    /// onto sub-goals by position and advances the active pointer accordingly.
    /// A step the model marks `done` that was `Active` triggers [`advance`]-like
    /// progression. Status strings are tolerant (reuse `PlanStatus::parse`
    /// semantics: "done"/"completed", "active"/"in_progress", else pending).
    pub fn apply_plan_statuses(&mut self, statuses: &[&str]) {
        for (i, raw) in statuses.iter().enumerate() {
            let Some(sg) = self.sub_goals.get_mut(i) else {
                break;
            };
            sg.status = parse_status(raw);
        }
        self.rederive_status();
    }

    /// Apply the executor's *evolving* plan (a `(description, status)` per step) to
    /// the goal, **bounded by the turn's anchor**: `turn_start_active` is the index
    /// of the sub-goal that was active when the turn started, and it is the only
    /// existing sub-goal a plan application may flip to `Done` — the drive works one
    /// milestone per turn, so a plan claiming more is self-certification, not
    /// progress. The anchor must come from the durable goal (stable across a turn),
    /// not the evolving proposal: repeated applications within one turn then share
    /// it and can't compound into a multi-step jump.
    ///
    /// Everything else the plan asserts is downgraded to evidence:
    /// - a `done` claim on any other non-done step records [`CLAIM_NOTE`] on it
    ///   (deduped), surfaced by [`prompt_section`](Self::prompt_section) when that
    ///   step becomes active;
    /// - a `pending`/`active` write onto a `Done` step is ignored with
    ///   [`REGRESSION_NOTE`] — plan updates never erase verified progress;
    /// - steps beyond the current list are **appended as `Pending`** regardless of
    ///   claimed status (a step born `Done` was the original bulk-completion bug),
    ///   so the plan still grows as the agent discovers work. By default there's
    ///   **no cap** — a user-set [`step_limit`](Self#structfield.step_limit) bounds
    ///   it. Existing sub-goals keep their richer planner descriptions.
    ///
    /// Then re-derive the overall status: completing the anchor activates the next
    /// not-done step; completing the last one finishes the goal.
    pub fn apply_plan(&mut self, steps: &[(String, GoalStatus)], turn_start_active: Option<usize>) {
        for (i, (description, status)) in steps.iter().enumerate() {
            if let Some(sg) = self.sub_goals.get_mut(i) {
                match (sg.status, *status) {
                    (GoalStatus::Done, GoalStatus::Done) => {}
                    (GoalStatus::Done, _) => push_note_deduped(sg, REGRESSION_NOTE),
                    (_, GoalStatus::Done) if Some(i) == turn_start_active => {
                        sg.status = GoalStatus::Done;
                    }
                    (_, GoalStatus::Done) => push_note_deduped(sg, CLAIM_NOTE),
                    // The cursor is owned by `rederive_status`; active/pending
                    // writes elsewhere are ignored.
                    _ => {}
                }
            } else if self
                .step_limit
                .is_none_or(|limit| self.sub_goals.len() < limit)
            {
                let mut sub_goal = SubGoal {
                    description: description.clone(),
                    status: GoalStatus::Pending,
                    attempts: 0,
                    notes: Vec::new(),
                    cap_continuations: 0,
                };
                if *status == GoalStatus::Done {
                    push_note_deduped(&mut sub_goal, CLAIM_NOTE);
                }
                self.sub_goals.push(sub_goal);
            }
        }
        self.rederive_status();
    }

    /// Continue past a sub-goal that just exhausted its retry budget: when
    /// drivable work remains (any `Pending` step), reactivate the goal — the
    /// exhausted step stays `Failed` as a visible scar, the first pending step
    /// becomes `Active`, and the drive keeps its momentum instead of one stuck
    /// milestone killing a mostly-done run. Returns `false` when nothing is
    /// left to drive (the goal stays `Failed` — that's the honest terminal
    /// state and the user's cue to intervene).
    pub fn continue_past_failure(&mut self) -> bool {
        if !self
            .sub_goals
            .iter()
            .any(|s| s.status == GoalStatus::Pending)
        {
            return false;
        }
        self.rederive_status();
        self.status == GoalStatus::Active
    }

    /// Append auditor-flagged milestones as `Pending` sub-goals, respecting
    /// `step_limit` and **deduplicating against every existing sub-goal** — an
    /// auditor re-flagging work the goal already tracks (done, failed, or
    /// pending) must not grow the plan. Then re-derive status, which
    /// reactivates the goal (the first pending step becomes active). Returns
    /// how many were actually appended; `0` means the audit converged (nothing
    /// new to add) or the step limit is saturated — either way the caller must
    /// finish rather than loop. Convergence-by-dedupe is the real audit-loop
    /// bound; the round cap is only a runaway guard.
    pub fn append_missing(&mut self, descriptions: &[String]) -> usize {
        let mut appended = 0;
        for description in descriptions {
            if self
                .step_limit
                .is_some_and(|limit| self.sub_goals.len() >= limit)
            {
                break;
            }
            let normalized = description.trim().to_ascii_lowercase();
            if self
                .sub_goals
                .iter()
                .any(|s| s.description.trim().to_ascii_lowercase() == normalized)
            {
                continue;
            }
            self.sub_goals.push(SubGoal {
                description: description.clone(),
                status: GoalStatus::Pending,
                attempts: 0,
                notes: Vec::new(),
                cap_continuations: 0,
            });
            appended += 1;
        }
        if appended > 0 {
            self.rederive_status();
        }
        appended
    }

    /// Re-derive the overall status from the sub-goals: `Done` iff all done;
    /// `Failed` iff a sub-goal failed and none is active; else `Active` — making the
    /// first not-done sub-goal active so there's always a cursor while in progress.
    fn rederive_status(&mut self) {
        if self.sub_goals.is_empty() {
            return;
        }
        if self.sub_goals.iter().all(|s| s.status == GoalStatus::Done) {
            self.status = GoalStatus::Done;
            return;
        }
        // Ensure the first not-done sub-goal is the active one (idempotent).
        let any_failed = self
            .sub_goals
            .iter()
            .any(|s| s.status == GoalStatus::Failed);
        for sg in &mut self.sub_goals {
            if sg.status == GoalStatus::Active {
                break;
            }
            if sg.status == GoalStatus::Pending {
                sg.status = GoalStatus::Active;
                break;
            }
        }
        if any_failed
            && !self
                .sub_goals
                .iter()
                .any(|s| s.status == GoalStatus::Active)
        {
            self.status = GoalStatus::Failed;
        } else {
            // Per the contract above: not all done, not failed-with-no-active
            // → Active. This must also revive a previously `Failed` goal whose
            // plan was revised to activate new work — leaving it `Failed`
            // would permanently disable auto-drive despite live sub-goals.
            self.status = GoalStatus::Active;
        }
    }

    /// Render the goal + sub-goal state as a system-prompt section, so the
    /// model resumes coherently each turn: the objective, the checklist with
    /// statuses, and any retry notes on the active sub-goal (so it doesn't
    /// repeat a failed approach). `None` when there are no sub-goals.
    pub fn prompt_section(&self) -> Option<String> {
        // A paused goal stops steering: no injection, so the agent treats the turn
        // as goal-free until `/goal resume`.
        if self.sub_goals.is_empty() || self.paused {
            return None;
        }
        let mut out = String::from(
            "\n\n[Long-horizon goal — work the active step, then advance only after validation]\n",
        );
        out.push_str(&format!("Objective: {}\n", self.objective));
        for (i, sg) in self.sub_goals.iter().enumerate() {
            let glyph = match sg.status {
                GoalStatus::Done => '✓',
                GoalStatus::Active => '▸',
                GoalStatus::Failed => '✗',
                GoalStatus::Pending => '○',
            };
            out.push_str(&format!("  {glyph} {}. {}\n", i + 1, sg.description));
            if sg.status == GoalStatus::Active && !sg.notes.is_empty() {
                out.push_str("     prior attempts (don't repeat these):\n");
                for n in &sg.notes {
                    out.push_str(&format!("       — {n}\n"));
                }
            }
        }
        out.push_str(
            "When calling update_plan, preserve this complete checklist in the same order, update its statuses, and append newly discovered implementation steps.\n",
        );
        Some(out)
    }
}

/// Append `note` to the sub-goal unless an identical note is already recorded —
/// the model tends to resubmit the same plan several times per turn, and one
/// claim/regression note per step is signal; five are noise.
fn push_note_deduped(sub_goal: &mut SubGoal, note: &str) {
    if !sub_goal.notes.iter().any(|n| n == note) {
        sub_goal.notes.push(note.to_string());
    }
}

/// Map a tolerant status string (from the model's `update_plan`) to a `GoalStatus`.
fn parse_status(raw: &str) -> GoalStatus {
    match raw.trim().to_ascii_lowercase().as_str() {
        "done" | "complete" | "completed" | "finished" => GoalStatus::Done,
        "active" | "in_progress" | "in-progress" | "doing" | "current" | "started" => {
            GoalStatus::Active
        }
        _ => GoalStatus::Pending,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn goal() -> Goal {
        Goal::new(
            "refactor the parser",
            vec![
                "write tests".into(),
                "rewrite parser".into(),
                "update callers".into(),
            ],
        )
    }

    #[test]
    fn new_goal_activates_first_sub_goal() {
        let g = goal();
        assert_eq!(g.status, GoalStatus::Active);
        assert_eq!(g.active_index(), Some(0));
        assert_eq!(g.sub_goals[1].status, GoalStatus::Pending);
    }

    #[test]
    fn advance_progresses_and_completes() {
        let mut g = goal();
        g.advance();
        assert_eq!(g.active_index(), Some(1));
        assert_eq!(g.sub_goals[0].status, GoalStatus::Done);
        g.advance();
        assert_eq!(g.active_index(), Some(2));
        g.advance();
        assert_eq!(g.status, GoalStatus::Done, "all done → goal done");
        assert_eq!(g.active_index(), None);
    }

    #[test]
    fn record_failure_retries_within_budget_then_fails() {
        let mut g = goal();
        // Budget 2: two failures still allow a retry; the third exhausts.
        assert!(g.record_failure("approach A didn't compile", 2));
        assert!(g.record_failure("approach B also failed", 2));
        assert!(
            !g.record_failure("third strike", 2),
            "budget exhausted → Failed"
        );
        assert_eq!(g.sub_goals[0].status, GoalStatus::Failed);
        assert_eq!(g.status, GoalStatus::Failed);
        assert_eq!(g.sub_goals[0].attempts, 3);
        assert_eq!(g.sub_goals[0].notes.len(), 3);
    }

    #[test]
    fn skip_active_advances_past_a_blocked_step() {
        let mut g = goal();
        g.skip_active("blocked on upstream API");
        assert_eq!(g.sub_goals[0].status, GoalStatus::Failed);
        assert_eq!(g.active_index(), Some(1), "advanced to next sub-goal");
        assert_eq!(g.status, GoalStatus::Active, "goal still active");
        // Skipping the *last* sub-goal fails the whole goal.
        g.skip_active("step 2 also blocked");
        assert_eq!(g.active_index(), Some(2), "advanced to last sub-goal");
        g.skip_active("last step blocked too");
        assert_eq!(
            g.status,
            GoalStatus::Failed,
            "skipping the last sub-goal fails the goal"
        );
    }

    #[test]
    fn apply_plan_statuses_maps_model_updates_and_advances() {
        let mut g = goal();
        // Model marks step 1 done, step 2 active, step 3 pending.
        g.apply_plan_statuses(&["done", "active", "pending"]);
        assert_eq!(g.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(g.sub_goals[1].status, GoalStatus::Active);
        assert_eq!(g.active_index(), Some(1));
        assert_ne!(g.status, GoalStatus::Done);
        // All done → goal done.
        g.apply_plan_statuses(&["done", "done", "done"]);
        assert_eq!(g.status, GoalStatus::Done);
    }

    #[test]
    fn apply_plan_statuses_tolerates_synonyms() {
        let mut g = goal();
        g.apply_plan_statuses(&["completed", "in_progress", "todo"]);
        assert_eq!(g.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(g.sub_goals[1].status, GoalStatus::Active);
        assert_eq!(g.sub_goals[2].status, GoalStatus::Pending);
    }

    #[test]
    fn prompt_section_lists_active_notes_so_model_doesnt_repeat() {
        let mut g = goal();
        g.record_failure("approach A", 2);
        let section = g.prompt_section().expect("nonempty goal renders");
        assert!(
            section.contains("refactor the parser"),
            "objective: {section}"
        );
        assert!(section.contains("▸"), "active glyph: {section}");
        assert!(
            section.contains("don't repeat these"),
            "retry-notes header: {section}"
        );
        assert!(section.contains("approach A"), "the note itself: {section}");
    }

    #[test]
    fn prompt_section_none_for_empty_goal() {
        let g = Goal::new("nothing", vec![]);
        assert!(g.prompt_section().is_none());
    }

    #[test]
    fn paused_goal_stops_steering_but_keeps_progress() {
        let mut g = goal();
        g.advance(); // sub-goal 2 active, 1 done
        g.paused = true;
        assert!(
            g.prompt_section().is_none(),
            "a paused goal is dropped from the system prompt"
        );
        // Progress is retained across the pause.
        assert_eq!(g.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(g.active_index(), Some(1));
        // Resume re-injects with progress intact.
        g.paused = false;
        let section = g.prompt_section().expect("resumed goal steers again");
        assert!(section.contains("refactor the parser"));
    }

    #[test]
    fn apply_plan_statuses_preserves_paused_flag() {
        let mut g = goal();
        g.paused = true;
        g.apply_plan_statuses(&["done", "active", "pending"]);
        assert!(g.paused, "re-deriving status must not clear the pause flag");
    }

    #[test]
    fn apply_plan_grows_as_the_agent_discovers_work() {
        let mut g = goal(); // 3 planner sub-goals
        let anchor = g.active_index();
        // The executor reports the 3, then discovers 2 more mid-project.
        g.apply_plan(
            &[
                ("write tests".into(), GoalStatus::Done),
                ("rewrite parser".into(), GoalStatus::Active),
                ("update callers".into(), GoalStatus::Pending),
                (
                    "fix a regression the rewrite surfaced".into(),
                    GoalStatus::Pending,
                ),
                ("update the changelog".into(), GoalStatus::Pending),
            ],
            anchor,
        );
        assert_eq!(g.sub_goals.len(), 5, "two discovered steps appended");
        assert_eq!(
            g.sub_goals[3].description,
            "fix a regression the rewrite surfaced"
        );
        assert_eq!(g.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(g.active_index(), Some(1));
    }

    #[test]
    fn apply_plan_keeps_planner_descriptions_for_existing_steps() {
        let mut g = goal();
        let anchor = g.active_index();
        // A terser executor title must not overwrite the planner's richer text.
        g.apply_plan(&[("wt".into(), GoalStatus::Done)], anchor);
        assert_eq!(g.sub_goals[0].description, "write tests");
        assert_eq!(g.sub_goals[0].status, GoalStatus::Done);
    }

    #[test]
    fn apply_plan_rejects_bulk_done_beyond_anchor() {
        let mut g = Goal::new(
            "big objective",
            (0..5).map(|i| format!("step {i}")).collect(),
        );
        let anchor = g.active_index();
        let all_done: Vec<(String, GoalStatus)> = (0..5)
            .map(|i| (format!("step {i}"), GoalStatus::Done))
            .collect();
        g.apply_plan(&all_done, anchor);
        assert_eq!(g.sub_goals[0].status, GoalStatus::Done, "anchor may finish");
        assert_eq!(g.active_index(), Some(1), "next step activated");
        for sg in &g.sub_goals[2..] {
            assert_eq!(sg.status, GoalStatus::Pending, "no teleporting to Done");
            assert_eq!(sg.attempts, 0);
            assert_eq!(sg.notes, vec![CLAIM_NOTE.to_string()], "claim recorded");
        }
        assert_eq!(
            g.sub_goals[1].notes,
            vec![CLAIM_NOTE.to_string()],
            "the now-active step keeps its claim note for the next drive turn"
        );
        assert_eq!(g.status, GoalStatus::Active, "goal is NOT done");
        assert!(g.should_auto_drive(), "the drive must keep going");
    }

    #[test]
    fn apply_plan_allows_single_step_advance() {
        let mut g = goal();
        g.apply_plan(
            &[
                ("write tests".into(), GoalStatus::Done),
                ("rewrite parser".into(), GoalStatus::Active),
                ("update callers".into(), GoalStatus::Pending),
            ],
            g.active_index(),
        );
        assert_eq!(g.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(g.active_index(), Some(1));
        assert_eq!(g.sub_goals[2].status, GoalStatus::Pending);
        assert!(
            g.sub_goals.iter().all(|s| s.notes.is_empty()),
            "an honest single-step advance records no notes"
        );
    }

    #[test]
    fn apply_plan_same_turn_calls_do_not_compound() {
        let mut g = goal();
        // Anchor is captured once per turn from the durable goal; both
        // applications use it even though the first advanced the proposal.
        let anchor = g.active_index();
        g.apply_plan(
            &[
                ("write tests".into(), GoalStatus::Done),
                ("rewrite parser".into(), GoalStatus::Active),
                ("update callers".into(), GoalStatus::Pending),
            ],
            anchor,
        );
        g.apply_plan(
            &[
                ("write tests".into(), GoalStatus::Done),
                ("rewrite parser".into(), GoalStatus::Done),
                ("update callers".into(), GoalStatus::Active),
            ],
            anchor,
        );
        assert_eq!(g.active_index(), Some(1), "still one step per turn");
        assert_eq!(g.sub_goals[1].status, GoalStatus::Active);
        assert_eq!(
            g.sub_goals[1].notes,
            vec![CLAIM_NOTE.to_string()],
            "the second done-claim became a note"
        );
        assert_eq!(g.sub_goals[2].status, GoalStatus::Pending);
    }

    #[test]
    fn apply_plan_claim_notes_dedupe_across_calls() {
        let mut g = goal();
        let anchor = g.active_index();
        let all_done: Vec<(String, GoalStatus)> = g
            .sub_goals
            .iter()
            .map(|s| (s.description.clone(), GoalStatus::Done))
            .collect();
        g.apply_plan(&all_done, anchor);
        g.apply_plan(&all_done, anchor);
        for sg in &g.sub_goals[1..] {
            assert_eq!(sg.notes.len(), 1, "one claim note despite two applies");
        }
    }

    #[test]
    fn apply_plan_appends_are_always_pending() {
        let mut g = goal();
        g.apply_plan(
            &[
                ("write tests".into(), GoalStatus::Active),
                ("rewrite parser".into(), GoalStatus::Pending),
                ("update callers".into(), GoalStatus::Pending),
                ("newly discovered step".into(), GoalStatus::Done),
            ],
            g.active_index(),
        );
        let appended = &g.sub_goals[3];
        assert_eq!(appended.status, GoalStatus::Pending, "no step born Done");
        assert_eq!(
            appended.notes,
            vec![CLAIM_NOTE.to_string()],
            "its done-claim survives as a note"
        );
    }

    #[test]
    fn apply_plan_ignores_regression_of_done_step() {
        let mut g = goal();
        g.advance(); // step 0 Done, step 1 Active
        let anchor = g.active_index();
        let revert = vec![
            ("write tests".into(), GoalStatus::Pending),
            ("rewrite parser".into(), GoalStatus::Active),
            ("update callers".into(), GoalStatus::Pending),
        ];
        g.apply_plan(&revert, anchor);
        g.apply_plan(&revert, anchor);
        assert_eq!(
            g.sub_goals[0].status,
            GoalStatus::Done,
            "Done never reverts"
        );
        assert_eq!(
            g.sub_goals[0].notes,
            vec![REGRESSION_NOTE.to_string()],
            "revert recorded once"
        );
        assert_eq!(g.active_index(), Some(1));
    }

    #[test]
    fn apply_plan_completing_last_step_finishes_goal() {
        let mut g = Goal::new("small", vec!["a".into(), "b".into()]);
        g.advance(); // a Done, b Active
        g.apply_plan(
            &[
                ("a".into(), GoalStatus::Done),
                ("b".into(), GoalStatus::Done),
            ],
            g.active_index(),
        );
        assert_eq!(g.status, GoalStatus::Done, "legitimate completion works");
    }

    #[test]
    fn apply_plan_with_no_anchor_flips_nothing() {
        let mut g = Goal::new("small", vec!["a".into(), "b".into()]);
        g.advance();
        g.advance(); // all Done, goal Done, no Active step
        assert_eq!(g.active_index(), None);
        g.apply_plan(
            &[
                ("a".into(), GoalStatus::Pending),
                ("b".into(), GoalStatus::Done),
                ("late discovery".into(), GoalStatus::Done),
            ],
            None,
        );
        assert_eq!(g.sub_goals[0].status, GoalStatus::Done, "no revert");
        assert_eq!(g.sub_goals[1].status, GoalStatus::Done);
        assert_eq!(
            g.sub_goals[2].notes,
            vec![CLAIM_NOTE.to_string()],
            "appended step is not born Done; its claim survives as a note"
        );
        assert_eq!(
            g.active_index(),
            Some(2),
            "the append (Pending at birth) reactivates the goal via rederive"
        );
    }

    #[test]
    fn apply_plan_skips_failed_step_when_activating_next() {
        let mut g = goal();
        g.skip_active("blocked"); // step 0 Failed, step 1 Active
        g.apply_plan(
            &[
                ("write tests".into(), GoalStatus::Pending),
                ("rewrite parser".into(), GoalStatus::Done),
                ("update callers".into(), GoalStatus::Pending),
            ],
            g.active_index(),
        );
        assert_eq!(g.sub_goals[0].status, GoalStatus::Failed, "Failed stays");
        assert_eq!(g.sub_goals[1].status, GoalStatus::Done);
        assert_eq!(
            g.active_index(),
            Some(2),
            "activation skips the Failed step"
        );
    }

    #[test]
    fn append_missing_reactivates_and_respects_limit() {
        let mut g = Goal::new("small", vec!["a".into()]);
        g.advance();
        assert_eq!(g.status, GoalStatus::Done);
        let appended = g.append_missing(&["found gap".into(), "another gap".into()]);
        assert_eq!(appended, 2);
        assert_eq!(
            g.status,
            GoalStatus::Active,
            "audit findings reopen the goal"
        );
        assert_eq!(g.active_index(), Some(1));
        assert!(g.should_auto_drive());

        // Saturated step limit: nothing appended, goal stays finished.
        let mut capped = Goal::new("small", vec!["a".into()]);
        capped.step_limit = Some(1);
        capped.advance();
        assert_eq!(capped.append_missing(&["gap".into()]), 0);
        assert_eq!(capped.status, GoalStatus::Done);
    }

    #[test]
    fn append_missing_dedupes_against_existing_sub_goals() {
        // Convergence: an auditor re-flagging work the goal already tracks
        // (any status, case/whitespace-insensitively) appends nothing — the
        // audit loop terminates by dedupe, not by burning its round cap.
        let mut g = Goal::new("small", vec!["Implement the exporter".into()]);
        g.advance();
        let appended = g.append_missing(&[
            "  implement THE exporter ".into(), // dup of the done step
            "Implement the importer".into(),    // genuinely new
            "Implement the importer".into(),    // dup within the batch
        ]);
        assert_eq!(appended, 1, "only the genuinely new milestone lands");
        assert_eq!(g.sub_goals.len(), 2);
        assert_eq!(g.status, GoalStatus::Active);

        // A fully repetitive round converges to zero.
        assert_eq!(g.append_missing(&["implement the importer".into()]), 0);
    }

    #[test]
    fn continue_past_failure_skips_when_pending_work_remains() {
        let mut g = goal();
        // Exhaust the first step's budget → goal Failed.
        g.record_failure("a", 0);
        assert_eq!(g.status, GoalStatus::Failed);
        assert!(g.continue_past_failure(), "pending steps → keep driving");
        assert_eq!(g.status, GoalStatus::Active);
        assert_eq!(g.sub_goals[0].status, GoalStatus::Failed, "scar stays");
        assert_eq!(g.active_index(), Some(1));
        assert!(g.should_auto_drive());

        // Nothing left to drive → stays Failed (honest terminal state).
        let mut done = Goal::new("small", vec!["a".into(), "b".into()]);
        done.advance(); // a Done, b Active
        done.record_failure("dead end", 0); // b Failed → goal Failed
        assert!(!done.continue_past_failure());
        assert_eq!(done.status, GoalStatus::Failed);
    }

    #[test]
    fn new_goal_defaults_team_on() {
        assert!(
            goal().team,
            "the skeptic gate is on by default for new goals"
        );
    }

    #[test]
    fn goal_without_new_fields_deserializes() {
        // A record shaped like pre-anchor/pre-audit sessions (no paused,
        // step_limit, team, skeptic_*, audit_rounds fields).
        let old = r#"{
            "objective": "port the service",
            "sub_goals": [
                {"description": "step one", "status": "Done"},
                {"description": "step two", "status": "Active"}
            ],
            "status": "Active"
        }"#;
        let g: Goal = serde_json::from_str(old).expect("old goal record loads");
        assert_eq!(g.audit_rounds, 0);
        assert!(!g.team);
        assert!(!g.paused);
        assert_eq!(g.active_index(), Some(1));
    }

    #[test]
    fn should_auto_drive_only_when_active_and_unpaused() {
        let mut g = goal();
        assert!(g.should_auto_drive(), "fresh goal drives");
        g.paused = true;
        assert!(!g.should_auto_drive(), "paused holds the drive");
        g.paused = false;
        g.advance();
        g.advance();
        g.advance();
        assert_eq!(g.status, GoalStatus::Done);
        assert!(!g.should_auto_drive(), "done goal stops driving");
        let mut failed = goal();
        failed.record_failure("a", 0);
        assert_eq!(failed.status, GoalStatus::Failed);
        assert!(!failed.should_auto_drive(), "failed goal stops driving");
        let empty = Goal::new("nothing", vec![]);
        assert!(!empty.should_auto_drive(), "empty goal never drives");
    }

    #[test]
    fn apply_plan_grows_without_limit_by_default() {
        let mut g = Goal::new("big", vec!["s0".into()]);
        let steps: Vec<(String, GoalStatus)> = (0..200)
            .map(|i| (format!("s{i}"), GoalStatus::Pending))
            .collect();
        g.apply_plan(&steps, g.active_index());
        assert_eq!(g.sub_goals.len(), 200, "no default cap — the plan expands");
    }

    #[test]
    fn apply_plan_respects_a_user_set_limit() {
        let mut g = Goal::new("big", vec!["s0".into()]);
        g.step_limit = Some(5);
        let steps: Vec<(String, GoalStatus)> = (0..50)
            .map(|i| (format!("s{i}"), GoalStatus::Pending))
            .collect();
        g.apply_plan(&steps, g.active_index());
        assert_eq!(g.sub_goals.len(), 5, "bounded by the user's /goal limit");
    }
}
