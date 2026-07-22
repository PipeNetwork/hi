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
    /// The reviewer judged the step unfixable-by-retry (contradiction or
    /// needs a user decision) — the driver skipped it with a visible scar.
    Escalated,
    Unavailable,
}

/// Why auto-drive is paused. Orthogonal to [`GoalStatus`]: a goal can be
/// `Active` with a pause reason (hold progress, stop drive) until resumed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalPauseReason {
    /// Not paused — drive may continue when status allows.
    #[default]
    None,
    /// User ran `/goal pause` (or equivalent).
    User,
    /// Frontend parked the drive after consecutive no-progress turns.
    Stall,
    /// Skeptic escalated / blocked further unattended advance.
    Skeptic,
    /// Infra failure (ledger/session/write) stopped the drive safely.
    Infra,
    /// Fresh plan awaiting human accept (`/goal resume` or `/goal accept`).
    Review,
}

impl GoalPauseReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::User => "user",
            Self::Stall => "stall",
            Self::Skeptic => "skeptic",
            Self::Infra => "infra",
            Self::Review => "review",
        }
    }

    pub fn is_paused(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// One capped history event for `/goal status` / postmortems.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalEvent {
    /// Unix seconds (best-effort).
    #[serde(default)]
    pub at: u64,
    /// Short machine tag: set, advance, fail, pause, resume, stall, skeptic, clear, edit, audit.
    pub kind: String,
    /// Human-readable detail.
    pub detail: String,
}

/// Cap on retained [`GoalEvent`]s (ring buffer).
pub const GOAL_EVENT_LIMIT: usize = 48;

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
    /// How many turns hit the per-turn step cap as continuations of this
    /// sub-goal (the milestone is bigger than one turn). Such turns don't burn
    /// the retry budget until the safety ceiling [`MAX_CAP_CONTINUATIONS`].
    /// Incrementing also marks the goal as changed, which keeps the frontend
    /// drive-stall counter from parking a long multi-turn milestone.
    /// `#[serde(default)]`.
    #[serde(default)]
    pub cap_continuations: u32,
    /// Consecutive capped turns that changed no files ("barren" caps). A capped
    /// turn that lands edits is real progress and resets this to 0; a run of
    /// barren caps means the milestone is genuinely stuck (the model can't land
    /// edits), so past [`MAX_BARREN_CAPS`] it fails instead of continuing. This
    /// lets a large milestone span many *productive* turns while still catching a
    /// thrashing one. `#[serde(default)]`.
    #[serde(default)]
    pub barren_caps: u32,
    /// How many rounds of on-the-fly decomposition produced this sub-goal. A
    /// milestone that keeps hitting the step cap while making progress is too big
    /// for one turn and is split into turn-sized sub-steps ([`Goal::decompose_active`]);
    /// children carry `split_depth + 1`, so recursion is bounded by
    /// [`MAX_SPLIT_DEPTH`]. `#[serde(default)]`.
    #[serde(default)]
    pub split_depth: u32,
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
    ///
    /// Prefer [`Self::pause_reason`] for new code; this bool stays in sync for
    /// older readers and session JSON.
    #[serde(default)]
    pub paused: bool,
    /// Typed pause reason. When non-[`GoalPauseReason::None`], [`Self::paused`]
    /// is also true. Unknown/missing on load → derived from `paused` (user).
    #[serde(default)]
    pub pause_reason: GoalPauseReason,
    /// Optional user-set ceiling on how many sub-goals the plan may grow to (via
    /// `/goal limit <n>`). `None` (the default) means **no limit** — the plan keeps
    /// expanding as the agent discovers work, which is the point for long,
    /// adventurous objectives ("port this service to Rust"). Part of the contract,
    /// so it persists with the goal. `#[serde(default)]` for older saved goals.
    #[serde(default)]
    pub step_limit: Option<usize>,
    /// Append-only (capped) ops log for status/postmortems.
    #[serde(default)]
    pub events: Vec<GoalEvent>,
    /// When true, completion audit has accepted the objective (or been skipped
    /// after max rounds). Surface-only; drive still uses status/sub-goals.
    #[serde(default)]
    pub objective_complete: bool,
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
    /// How many steps the reviewer escalated as unfixable-by-retry (skipped
    /// with a scar instead of burning the retry budget). `#[serde(default)]`.
    #[serde(default)]
    pub skeptic_escalations: u32,
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
sub-goal now. Favor concrete implementation over exploration — you have context from earlier \
turns, so write and edit the code that delivers this sub-goal rather than re-reading the repo; \
land real file changes this turn. Then call update_plan with the full goal checklist in its \
existing order — update statuses and append any newly discovered implementation steps.";

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

/// Note added when a milestone hit the step cap without landing any file edits,
/// so the next turn's system prompt steers the model to implement rather than
/// keep exploring. Deduped, so it appears once.
pub const BARREN_CAP_NOTE: &str = "a prior turn hit the step cap while exploring without landing \
any file edits — this turn, make concrete code changes (write/edit the files this sub-goal needs) \
instead of more reading";

/// Safety ceiling on how many step-capped turns a single sub-goal may span
/// before capped turns start burning its retry budget again — a runaway guard,
/// not the real gate (that's [`MAX_BARREN_CAPS`]). A big milestone (a whole
/// crate from scratch) legitimately spans many capped turns as long as it keeps
/// landing edits; only a milestone that keeps capping out *without* progress, or
/// one that blows this generous ceiling, is judged by the retry/skip machinery.
pub const MAX_CAP_CONTINUATIONS: u32 = 40;

/// Consecutive capped turns that change no files before a milestone is judged
/// stuck (rather than merely large). A capped turn that lands edits resets the
/// count, so a productive milestone spans as many turns as it needs; a run of
/// this many barren caps means the model can't make progress on it.
pub const MAX_BARREN_CAPS: u32 = 3;

/// A clean-success turn whose net change is at most this many bytes skips the
/// `/goal team` skeptic review: the defect classes the gate exists to catch
/// (stub stand-ins, wrong-artifact substitutions, explicitly-required cases
/// left unhandled) can't hide in a few bytes of diff, and verify already
/// passed. Sized so a typo fix or a one-line tweak doesn't pay a second-model
/// round-trip, while any real implementation step still reviews.
pub const SKEPTIC_TRIVIAL_DIFF_BYTES: u64 = 64;

/// How many productive step-capped continuations a milestone takes before it's
/// judged too big for one turn and decomposed on the fly into turn-sized
/// sub-steps. Lower than [`MAX_CAP_CONTINUATIONS`] so a huge milestone is split
/// rather than ground out over dozens of turns.
pub const DECOMPOSE_AFTER_CONTINUATIONS: u32 = 4;

/// Maximum on-the-fly decomposition depth: a milestone split into sub-steps may
/// have its sub-steps split once more, but no deeper — a bound on recursion so a
/// pathological objective can't fan out without end.
pub const MAX_SPLIT_DEPTH: u32 = 2;

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
                barren_caps: 0,
                split_depth: 0,
            })
            .collect();
        let mut g = Self {
            objective: objective.into(),
            sub_goals,
            status: GoalStatus::Active,
            paused: false,
            pause_reason: GoalPauseReason::None,
            step_limit: None,
            events: Vec::new(),
            objective_complete: false,
            team: true,
            skeptic_objections: 0,
            skeptic_unavailable: 0,
            skeptic_escalations: 0,
            last_skeptic_status: None,
            audit_rounds: 0,
        };
        g.push_event("set", "goal created");
        g
    }

    /// Whether frontends should keep auto-driving this goal between turns: it's
    /// still in progress, not paused by the user, and actually has steps. `Done`,
    /// `Failed`, paused, or empty goals are left alone.
    pub fn should_auto_drive(&self) -> bool {
        self.status == GoalStatus::Active
            && !self.is_paused()
            && !self.sub_goals.is_empty()
            && !self.objective_complete
    }

    /// Effective pause: prefers typed reason; falls back to legacy `paused` bool.
    pub fn is_paused(&self) -> bool {
        self.pause_reason.is_paused() || self.paused
    }

    /// Pause with a typed reason (keeps `paused` in sync).
    pub fn pause(&mut self, reason: GoalPauseReason) {
        if matches!(reason, GoalPauseReason::None) {
            self.resume();
            return;
        }
        self.pause_reason = reason;
        self.paused = true;
        self.push_event("pause", format!("reason={}", reason.as_str()));
    }

    /// Clear any pause so auto-drive may continue.
    pub fn resume(&mut self) {
        if self.is_paused() {
            let prev = if self.pause_reason.is_paused() {
                self.pause_reason.as_str()
            } else {
                "user"
            };
            self.push_event("resume", format!("cleared {prev} pause"));
        }
        self.pause_reason = GoalPauseReason::None;
        self.paused = false;
    }

    /// Append a capped history event.
    pub fn push_event(&mut self, kind: impl Into<String>, detail: impl Into<String>) {
        let at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.events.push(GoalEvent {
            at,
            kind: kind.into(),
            detail: detail.into(),
        });
        if self.events.len() > GOAL_EVENT_LIMIT {
            let drop_n = self.events.len() - GOAL_EVENT_LIMIT;
            self.events.drain(0..drop_n);
        }
    }

    /// Rich multi-line status for `/goal` / `/goal status`.
    pub fn status_report(&self) -> String {
        let done = self
            .sub_goals
            .iter()
            .filter(|s| s.status == GoalStatus::Done)
            .count();
        let total = self.sub_goals.len();
        let active = self
            .active_sub_goal()
            .map(|s| s.description.as_str())
            .unwrap_or("(none)");
        let pause = if self.is_paused() {
            let reason = if self.pause_reason.is_paused() {
                self.pause_reason.as_str()
            } else {
                "user"
            };
            format!("paused ({reason})")
        } else {
            "running".into()
        };
        let limit = self
            .step_limit
            .map(|n| n.to_string())
            .unwrap_or_else(|| "none".into());
        let mut out = String::new();
        out.push_str(&format!("goal: {}\n", self.objective));
        out.push_str(&format!(
            "  state: {:?} · drive: {pause} · steps: {done}/{total} done · limit: {limit}\n",
            self.status
        ));
        out.push_str(&format!("  active: {active}\n"));
        out.push_str(&format!(
            "  team: {} · skeptic last: {} · objections: {} · unavailable: {} · escalations: {}\n",
            if self.team { "on" } else { "off" },
            self.last_skeptic_status
                .map(|s| format!("{s:?}"))
                .unwrap_or_else(|| "not run".into()),
            self.skeptic_objections,
            self.skeptic_unavailable,
            self.skeptic_escalations,
        ));
        out.push_str(&format!(
            "  completion audit rounds: {} · objective_complete: {}\n",
            self.audit_rounds, self.objective_complete
        ));
        if !self.events.is_empty() {
            out.push_str("  recent events:\n");
            for ev in self
                .events
                .iter()
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                out.push_str(&format!("    - {}: {}\n", ev.kind, ev.detail));
            }
        }
        // Compact checklist window around active.
        out.push_str("  checklist:\n");
        let ai = self.active_index().unwrap_or(0);
        let start = ai.saturating_sub(2);
        let end = (ai + 5).min(total);
        if start > 0 {
            out.push_str(&format!("    … {start} earlier step(s)\n"));
        }
        for (i, sg) in self
            .sub_goals
            .iter()
            .enumerate()
            .skip(start)
            .take(end - start)
        {
            let mark = match sg.status {
                GoalStatus::Done => "x",
                GoalStatus::Active => ">",
                GoalStatus::Pending => " ",
                GoalStatus::Failed => "!",
            };
            out.push_str(&format!("    {:>2}. [{mark}] {}\n", i + 1, sg.description));
        }
        if end < total {
            out.push_str(&format!("    … {} later step(s)\n", total - end));
        }
        out.push_str(
            "  commands: /goal pause|resume|accept|status|edit …|limit …|team …|clear|export\n",
        );
        out
    }

    /// Markdown snapshot for human review (export-only; struct remains SoT).
    pub fn to_markdown(&self) -> String {
        let mut out = format!("# Goal\n\n**Objective:** {}\n\n", self.objective);
        out.push_str(&format!(
            "- status: {:?}\n- drive: {}\n- team: {}\n\n## Checklist\n\n",
            self.status,
            if self.is_paused() {
                self.pause_reason.as_str()
            } else {
                "running"
            },
            if self.team { "on" } else { "off" },
        ));
        for (i, sg) in self.sub_goals.iter().enumerate() {
            let box_ = match sg.status {
                GoalStatus::Done => "[x]",
                GoalStatus::Active => "[>]",
                GoalStatus::Failed => "[!]",
                GoalStatus::Pending => "[ ]",
            };
            out.push_str(&format!("{}. {} {}\n", i + 1, box_, sg.description));
            for n in &sg.notes {
                out.push_str(&format!("   - note: {n}\n"));
            }
        }
        if !self.events.is_empty() {
            out.push_str("\n## Events\n\n");
            for ev in &self.events {
                out.push_str(&format!("- `{}`: {}\n", ev.kind, ev.detail));
            }
        }
        out
    }

    /// Write export-only markdown next to the workspace `.hi/` dir.
    pub fn export_markdown_to(
        &self,
        workspace: &std::path::Path,
    ) -> std::io::Result<std::path::PathBuf> {
        let dir = workspace.join(".hi");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("goal-plan.md");
        std::fs::write(&path, self.to_markdown())?;
        Ok(path)
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
            let done_desc = self.sub_goals[i].description.clone();
            self.sub_goals[i].status = GoalStatus::Done;
            if let Some(next) = self.sub_goals.get_mut(i + 1) {
                next.status = GoalStatus::Active;
                let next_desc = next.description.clone();
                self.push_event(
                    "advance",
                    format!(
                        "completed step {}: {}; active → {}",
                        i + 1,
                        done_desc,
                        next_desc
                    ),
                );
            } else {
                self.status = GoalStatus::Done;
                self.push_event(
                    "advance",
                    format!("completed final step {}: {}; goal Done", i + 1, done_desc),
                );
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
                    barren_caps: 0,
                    split_depth: 0,
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
                barren_caps: 0,
                split_depth: 0,
            });
            appended += 1;
        }
        if appended > 0 {
            self.rederive_status();
        }
        appended
    }

    /// Split the active sub-goal in place into `sub_steps`: the active milestone
    /// is too big for one turn, so it's replaced by turn-sized sub-steps (Pending,
    /// carrying `split_depth + 1` so recursion is bounded by [`MAX_SPLIT_DEPTH`]).
    /// Returns the number spliced in, or `0` (a no-op) if there's no active
    /// sub-goal, fewer than two usable sub-steps (splitting into one is
    /// pointless), or the split would exceed `step_limit`. Re-derives status so
    /// the first sub-step becomes active.
    pub fn decompose_active(&mut self, sub_steps: &[String]) -> usize {
        let Some(active) = self.active_index() else {
            return 0;
        };
        let parent_depth = self.sub_goals[active].split_depth;
        let mut seen: Vec<String> = Vec::new();
        let mut children: Vec<SubGoal> = Vec::new();
        for d in sub_steps {
            let d = d.trim();
            if d.is_empty() {
                continue;
            }
            let norm = d.to_ascii_lowercase();
            if seen.contains(&norm) {
                continue;
            }
            seen.push(norm);
            children.push(SubGoal {
                description: d.to_string(),
                status: GoalStatus::Pending,
                attempts: 0,
                notes: Vec::new(),
                cap_continuations: 0,
                barren_caps: 0,
                split_depth: parent_depth + 1,
            });
        }
        if children.len() < 2 {
            return 0;
        }
        if let Some(limit) = self.step_limit
            && self.sub_goals.len() + children.len() - 1 > limit
        {
            return 0;
        }
        let n = children.len();
        self.sub_goals.splice(active..=active, children);
        self.rederive_status();
        n
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
        if self.sub_goals.is_empty() || self.is_paused() {
            return None;
        }
        let mut out = String::from(
            "\n\n[Long-horizon goal — work the active step, then advance only after validation]\n",
        );
        out.push_str(&format!("Objective: {}\n", self.objective));
        // The full checklist rides in the system prompt every turn; on a long
        // goal (the planner may produce 120 milestones) re-rendering every line
        // is the dominant per-turn token cost — and it busts provider prefix
        // caches on each status flip. The model works one step at a time, so
        // compact completed runs and only expand a window around the active
        // step: the near past (what it just did), the active step with its
        // retry notes, and the near future (what's next). The completion
        // auditor renders the full checklist itself, so nothing is lost.
        let total = self.sub_goals.len();
        let active = self.active_index();
        let (window_start, window_end) = match active {
            Some(i) => (i.saturating_sub(2), (i + 4).min(total)),
            // No active step (all done / all failed): show the last few steps
            // so the model sees what the drive landed on.
            None => (total.saturating_sub(6), total),
        };
        let mut i = 0;
        let mut pending_skipped = false;
        while i < total {
            let in_window = i >= window_start && i < window_end;
            if !in_window {
                match self.sub_goals[i].status {
                    GoalStatus::Done => {
                        // Collapse a run of completed steps into one line.
                        let mut end = i + 1;
                        while end < total
                            && !(end >= window_start && end < window_end)
                            && self.sub_goals[end].status == GoalStatus::Done
                        {
                            end += 1;
                        }
                        if end == i + 1 {
                            out.push_str(&format!(
                                "  ✓ {}. {}\n",
                                i + 1,
                                self.sub_goals[i].description
                            ));
                        } else {
                            out.push_str(&format!("  ✓ {}–{} completed\n", i + 1, end));
                        }
                        i = end;
                    }
                    _ => {
                        // A single compact marker for the not-yet-reached tail —
                        // not one line per pending step.
                        if !pending_skipped {
                            let remaining = (i..total)
                                .filter(|&j| !(j >= window_start && j < window_end))
                                .count();
                            if remaining > 0 {
                                out.push_str(&format!(
                                    "  … {remaining} more step(s) after the window\n"
                                ));
                            }
                            pending_skipped = true;
                        }
                        i += 1;
                    }
                }
                continue;
            }
            let sg = &self.sub_goals[i];
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
            i += 1;
        }
        out.push_str(
            "When calling update_plan, resubmit the full checklist in its existing order (including steps compacted above), update statuses, and append newly discovered implementation steps.\n",
        );
        Some(out)
    }
}

/// Append `note` to the sub-goal unless an identical note is already recorded —
/// the model tends to resubmit the same plan several times per turn, and one
/// claim/regression note per step is signal; five are noise.
pub(crate) fn push_note_deduped(sub_goal: &mut SubGoal, note: &str) {
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
    fn prompt_section_compacts_done_runs_and_pending_tail() {
        // A long goal partway through: many done, an active step, many pending.
        let steps: Vec<String> = (1..=20).map(|i| format!("milestone {i}")).collect();
        let mut g = Goal::new("big refactor", steps);
        for _ in 0..9 {
            g.advance(); // milestones 1–9 done, 10 active
        }
        let section = g.prompt_section().expect("renders");
        // The leading done run is collapsed, not listed line-by-line.
        assert!(
            section.contains("✓ 1–7 completed"),
            "done run compacted: {section}"
        );
        // The window shows the two steps before the active one.
        assert!(section.contains("✓ 8. milestone 8"), "near past: {section}");
        assert!(section.contains("✓ 9. milestone 9"), "near past: {section}");
        assert!(
            section.contains("▸ 10. milestone 10"),
            "active step: {section}"
        );
        // The next three pending steps are visible, the tail is summarized.
        assert!(
            section.contains("○ 11. milestone 11"),
            "near future: {section}"
        );
        assert!(
            section.contains("○ 13. milestone 13"),
            "near future: {section}"
        );
        assert!(
            !section.contains("milestone 14"),
            "tail is compacted: {section}"
        );
        assert!(
            section.contains("7 more step(s)"),
            "tail summary: {section}"
        );
        // No individual lines for the compacted done run.
        assert!(
            !section.contains("milestone 3"),
            "compacted done step absent: {section}"
        );
    }

    #[test]
    fn prompt_section_short_goal_renders_every_step() {
        // Small goals fit the window entirely — nothing is compacted.
        let mut g = goal(); // 3 sub-goals
        g.advance();
        let section = g.prompt_section().expect("renders");
        assert!(section.contains("✓ 1. write tests"), "{section}");
        assert!(section.contains("▸ 2. rewrite parser"), "{section}");
        assert!(section.contains("○ 3. update callers"), "{section}");
        assert!(!section.contains("more step(s)"), "{section}");
    }

    #[test]
    fn prompt_section_finished_goal_compacts_all_but_last_done() {
        let steps: Vec<String> = (1..=10).map(|i| format!("step {i}")).collect();
        let mut g = Goal::new("done goal", steps);
        for _ in 0..10 {
            g.advance();
        }
        assert_eq!(g.status, GoalStatus::Done);
        let section = g.prompt_section().expect("renders");
        // No active step: the window shows the last few steps, the rest
        // collapses into one compacted run.
        assert!(section.contains("✓ 1–4 completed"), "{section}");
        assert!(section.contains("✓ 5. step 5"), "window tail: {section}");
        assert!(section.contains("✓ 10. step 10"), "window tail: {section}");
        assert!(!section.contains("step 2."), "compacted: {section}");
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
    fn decompose_active_replaces_the_milestone_with_substeps() {
        let mut g = Goal::new("build", vec!["big crate".into(), "next".into()]);
        // The active milestone (index 0) splits into three turn-sized sub-steps.
        let n = g.decompose_active(&[
            "scaffold the crate".into(),
            "implement the core".into(),
            "add tests".into(),
        ]);
        assert_eq!(n, 3);
        assert_eq!(
            g.sub_goals.len(),
            4,
            "3 sub-steps replace 1 milestone, + next"
        );
        assert_eq!(g.sub_goals[0].description, "scaffold the crate");
        assert_eq!(
            g.sub_goals[0].status,
            GoalStatus::Active,
            "first sub-step active"
        );
        assert_eq!(g.sub_goals[0].split_depth, 1, "children carry depth+1");
        assert_eq!(
            g.sub_goals[3].description, "next",
            "the rest of the plan is preserved"
        );
        assert_eq!(g.status, GoalStatus::Active);

        // Fewer than two usable sub-steps is a no-op (splitting into one is pointless).
        let before = g.sub_goals.len();
        assert_eq!(g.decompose_active(&["only one".into()]), 0);
        assert_eq!(
            g.decompose_active(&["a".into(), "  ".into(), "A".into()]),
            0
        );
        assert_eq!(g.sub_goals.len(), before);

        // A step limit that the split would exceed blocks it.
        let mut capped = Goal::new("build", vec!["big".into()]);
        capped.step_limit = Some(2);
        assert_eq!(
            capped.decompose_active(&["a".into(), "b".into(), "c".into()]),
            0,
            "split past the step limit is refused"
        );
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
        assert_eq!(g.skeptic_escalations, 0);
        assert_eq!(g.sub_goals[0].cap_continuations, 0);
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
