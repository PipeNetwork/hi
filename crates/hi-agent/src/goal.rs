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
    /// Cannot proceed here for a reason no retry changes — a prerequisite the
    /// environment doesn't have (a database that isn't running, a binary that
    /// isn't installed, a credential that wasn't provided).
    ///
    /// Distinct from [`Self::Failed`] on purpose. A failed step was judged and
    /// found wanting; a blocked step was never judgeable, and reporting it as a
    /// failure both slanders the work and hides the one thing the user could
    /// act on. Blocked steps cost no retry budget and are listed with their
    /// missing prerequisite so the user can satisfy it and resume.
    Blocked,
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
    /// The turn budget (`/goal budget <n>`) was spent. Progress is intact and
    /// `/goal resume` continues — this is a checkpoint, not a failure.
    Budget,
    /// A confirm could not be answered interactively and was parked in the
    /// approval inbox (`/inbox`). Drive waits until the item is resolved.
    Approval,
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
            Self::Budget => "budget",
            Self::Approval => "approval",
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

/// One step of a decomposed goal. `attempts` retains retry history; ordinary
/// goals do not turn that diagnostic count into a finite execution ceiling.
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
    /// How many turns hit an explicit finite per-turn work cap (model steps or
    /// tool calls) as continuations of this sub-goal (the milestone is bigger
    /// than one turn). Such turns don't burn the retry budget until the safety
    /// count. Productive capped turns are not subject to a lifetime ceiling;
    /// only consecutive barren caps are bounded by [`MAX_BARREN_CAPS`].
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
    /// milestone that keeps hitting a work cap while making progress is too big
    /// for one turn and is split into turn-sized sub-steps ([`Goal::decompose_active`]);
    /// children carry `split_depth + 1`, so recursion is bounded by
    /// [`MAX_SPLIT_DEPTH`]. `#[serde(default)]`.
    #[serde(default)]
    pub split_depth: u32,
    /// Consecutive turns on this sub-goal that ended without verification
    /// reaching a verdict — the checks timed out, or the harness around them
    /// failed. These are **not** attempts: nothing judged the work, so charging
    /// them to the retry budget marks a step `Failed` for a defect in the
    /// environment. Reset by any turn that does reach a verdict; past
    /// [`MAX_UNJUDGED_TURNS`] the drive parks for the user instead of grinding.
    /// `#[serde(default)]`.
    #[serde(default)]
    pub unjudged_turns: u32,
    /// Turns that ended with verified workspace changes while this sub-goal
    /// stayed active — real progress that did not finish the milestone.
    ///
    /// Distinct from [`Self::cap_continuations`], which only counts turns that
    /// exhausted an explicit per-turn work budget. A model that ends its turns cleanly never
    /// trips that, so an oversized milestone could consume turns indefinitely
    /// without any signal that it should be split. Past
    /// [`DECOMPOSE_AFTER_PRODUCTIVE_TURNS`] the milestone is decomposed.
    /// Reset when the step completes or is otherwise resolved.
    /// `#[serde(default)]`.
    #[serde(default)]
    pub productive_turns: u32,
    /// Set when this step was skipped because goal-drive stalled (four
    /// no-progress turns). Eligible for one second pass after the first sweep
    /// if anything else completed. `#[serde(default)]`.
    #[serde(default)]
    pub stall_skipped: bool,
    /// This stall-skipped step was already returned to `Pending` for a second
    /// pass. A second stall stays `Failed`. `#[serde(default)]`.
    #[serde(default)]
    pub requeued: bool,
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
    /// Sub-goals abandoned in a row without a single completion in between.
    ///
    /// Skipping past an exhausted step keeps a long run alive when one
    /// milestone is stuck, but nothing bounded how far it could walk: a goal
    /// whose *every* step fails still advances its cursor, so thrashing and
    /// progress look identical from the outside. Reset by any completion; past
    /// [`MAX_SKIPS_WITHOUT_COMPLETION`] with nothing yet done, the drive parks.
    /// `#[serde(default)]`.
    #[serde(default)]
    pub consecutive_skips: u32,
    /// Optional user-set ceiling on how many drive turns this goal may consume
    /// before it parks and reports. `None` is the default: normal goal work
    /// continues until completion. `/goal budget <n>` installs a finite cap and
    /// `/goal budget off` removes it. `#[serde(default)]`.
    #[serde(default)]
    pub turn_budget: Option<u32>,
    /// Legacy marker for turn budgets generated automatically by older hi
    /// versions. New goals never set it; load/install paths clear both this
    /// marker and its generated budget so an old default cannot keep parking a
    /// resumed goal. Explicit `/goal budget` values always have this false.
    #[serde(default)]
    pub budget_auto: bool,
    /// Drive turns consumed so far, counted whether the turn succeeded or not —
    /// a turn that failed still spent time and tokens. `#[serde(default)]`.
    #[serde(default)]
    pub turns_spent: u32,
    /// How many steps the reviewer escalated as unfixable-by-retry (skipped
    /// with a scar instead of burning the retry budget). `#[serde(default)]`.
    #[serde(default)]
    pub skeptic_escalations: u32,
    #[serde(default)]
    pub last_skeptic_status: Option<SkepticStatus>,
    /// How many completion-audit rounds have appended missing work to this goal.
    /// This is diagnostic history, not a completion ceiling: repeated findings
    /// converge through deduplication, while genuinely new required work keeps
    /// the goal active. `#[serde(default)]` so older saved goals load at 0.
    #[serde(default)]
    pub audit_rounds: u32,
    /// When true, Goal-drive continues without a human at the overlay:
    /// confirms park in `/inbox` instead of elevating to Always. Opt-in via
    /// `/goal unattended on` or `/goal --unattended`. `#[serde(default)]`.
    #[serde(default)]
    pub unattended: bool,
}

/// Unlimited sentinel for ordinary per-sub-goal retries.
///
/// A failed verifier/reviewer round can still retain useful edits and expose the
/// next concrete issue, so a fixed attempt count must not terminate otherwise
/// progressing goal work. Callers may still pass an explicitly finite value to
/// [`Goal::record_failure`]; the independent drive no-progress guard remains the
/// circuit breaker for a step that is actually spinning.
pub const DEFAULT_SUBGOAL_RETRIES: u32 = u32::MAX;

/// Shown when `/goal unattended` is turned on.
pub const UNATTENDED_DRIVE_WARNING: &str = "goal unattended: confirms park in /inbox (permission mode stays Ask/Auto) until the goal ends, pauses, or /goal unattended off";

/// Consecutive turns a sub-goal may end unjudged (verification never reached a
/// verdict) before the drive parks for the user.
///
/// Two is enough to ride out a flaky one-off; beyond that the checks themselves
/// are the problem, and no amount of model effort fixes a verify command that
/// cannot finish. Parking surfaces it in seconds instead of burning hours.
pub const MAX_UNJUDGED_TURNS: u32 = 2;

/// How many sub-goals may be abandoned in a row *before any has completed*
/// before the drive parks.
///
/// A run that has never once finished a milestone is not making progress no
/// matter how far its cursor has moved. Two consecutive skips with zero
/// completions is the earliest point that is unambiguous rather than unlucky.
pub const MAX_SKIPS_WITHOUT_COMPLETION: u32 = 2;

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

/// Tool-call-cap counterpart to [`BARREN_CAP_NOTE`]. Kept distinct so a goal
/// resumed on the next turn is not told that its model-step budget ran out when
/// only an explicitly configured tool-execution budget did.
pub const BARREN_TOOL_CAP_NOTE: &str = "a prior turn hit the tool-call cap while exploring without \
landing any file edits — this turn, make concrete code changes (write/edit the files this sub-goal \
needs) instead of more reading";

/// Note recorded when both explicit per-turn caps fired together.
pub const BARREN_BOTH_CAPS_NOTE: &str = "a prior turn hit the step and tool-call caps while exploring \
without landing any file edits — this turn, make concrete code changes (write/edit the files this \
sub-goal needs) instead of more reading";

/// Unlimited sentinel retained for source/serialization compatibility with the
/// earlier capped-continuation API. Productive work-capped turns may span as
/// many turns as needed; [`MAX_BARREN_CAPS`] remains the no-progress circuit.
pub const MAX_CAP_CONTINUATIONS: u32 = u32::MAX;

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

/// Compatibility helper retained for callers compiled against the earlier
/// automatic-budget API. Automatic drive budgets are now unlimited.
pub fn auto_budget_for(_steps: usize) -> u32 {
    u32::MAX
}

/// How many productive explicitly work-capped continuations a milestone takes before it's
/// judged too big for one turn and decomposed on the fly into turn-sized
/// sub-steps. Decomposition improves milestone granularity; it is not a lifetime
/// ceiling, and a failed/unavailable planner leaves productive work drivable.
pub const DECOMPOSE_AFTER_CONTINUATIONS: u32 = 4;

/// How many turns a milestone may consume while *landing verified work* without
/// completing before it is judged too big and split into turn-sized sub-steps.
///
/// The explicit-cap signal above only fires when a turn runs out of configured
/// model rounds or tool calls — so an uncapped milestone sized in days
/// rather than turns could absorb turn after turn of real, verified progress and
/// still never reach `Done`, with nothing to distinguish it from a small step
/// being worked carefully. Productive-but-unfinished turns are that missing
/// signal: the work is landing, so the step is not stuck, it is simply too
/// large.
pub const DECOMPOSE_AFTER_PRODUCTIVE_TURNS: u32 = 3;

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
        let mut sub_goals = Vec::new();
        let mut known_descriptions = std::collections::HashSet::new();
        for description in sub_goal_descriptions {
            let Some(normalized) = normalized_description(&description) else {
                continue;
            };
            if !known_descriptions.insert(normalized) {
                continue;
            }
            let status = if sub_goals.is_empty() {
                GoalStatus::Active
            } else {
                GoalStatus::Pending
            };
            sub_goals.push(SubGoal {
                description: description.trim().to_string(),
                status,
                attempts: 0,
                notes: Vec::new(),
                cap_continuations: 0,
                barren_caps: 0,
                split_depth: 0,
                unjudged_turns: 0,
                productive_turns: 0,
                stall_skipped: false,
                requeued: false,
            });
        }
        let mut g = Self {
            objective: objective.into(),
            sub_goals,
            status: GoalStatus::Active,
            paused: false,
            pause_reason: GoalPauseReason::None,
            consecutive_skips: 0,
            turn_budget: None,
            budget_auto: false,
            turns_spent: 0,
            step_limit: None,
            events: Vec::new(),
            objective_complete: false,
            team: true,
            skeptic_objections: 0,
            skeptic_unavailable: 0,
            skeptic_escalations: 0,
            last_skeptic_status: None,
            audit_rounds: 0,
            unattended: false,
        };
        g.push_event("set", "goal created");
        g
    }

    /// Remove a generated turn ceiling written by an older hi version. A
    /// budget-pause caused solely by that obsolete default is reopened too;
    /// explicit user budgets have `budget_auto == false` and are untouched.
    pub(crate) fn clear_legacy_automatic_budget(&mut self) -> bool {
        if !self.budget_auto {
            return false;
        }
        self.budget_auto = false;
        self.turn_budget = None;
        if self.pause_reason == GoalPauseReason::Budget {
            self.resume();
        }
        true
    }

    /// Remaining structured-goal work, ignoring pause and stall. Parked and
    /// paused goals still have leftover work so empty Enter can resume them.
    pub fn has_drive_work(&self) -> bool {
        self.status == GoalStatus::Active && !self.sub_goals.is_empty() && !self.objective_complete
    }

    /// Whether frontends should keep auto-driving this goal between turns: it's
    /// still in progress, not paused by the user, and actually has steps. `Done`,
    /// `Failed`, paused, or empty goals are left alone.
    pub fn should_auto_drive(&self) -> bool {
        self.has_drive_work() && !self.is_paused()
    }

    /// Effective pause: prefers typed reason; falls back to legacy `paused` bool.
    pub fn is_paused(&self) -> bool {
        self.pause_reason.is_paused() || self.paused
    }

    /// Whether this goal changed structurally in a way that counts as progress
    /// for the frontend's auto-drive stall guard.
    ///
    /// Retry notes/counters, event history, reviewer diagnostics, and turn
    /// accounting are evidence *about* an attempt, not completed work. Ignoring
    /// them here is what lets the independent no-progress guard park an
    /// unlimited-default retry loop. File changes and novel read/search evidence
    /// are accounted separately by `goal_drive_made_progress` before this
    /// structural fallback runs.
    pub fn drive_state_changed_since(&self, previous: &Self) -> bool {
        let steps_changed = self.sub_goals.len() != previous.sub_goals.len()
            || self
                .sub_goals
                .iter()
                .zip(&previous.sub_goals)
                .any(|(current, prior)| {
                    current.description != prior.description || current.status != prior.status
                });
        self.objective != previous.objective
            || steps_changed
            || self.status != previous.status
            || self.paused != previous.paused
            || self.pause_reason != previous.pause_reason
            || self.objective_complete != previous.objective_complete
    }

    /// Pause with a typed reason (keeps `paused` in sync).
    pub fn pause(&mut self, reason: GoalPauseReason) {
        if matches!(reason, GoalPauseReason::None) {
            self.resume();
            return;
        }
        if self.paused && self.pause_reason == reason {
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

        // A wholly blocked goal has no `Active` cursor, so merely clearing a
        // pause would leave `/goal resume` inert forever. Re-activate the
        // earliest blocked step as a deliberate re-check: if the prerequisite
        // is still absent, the turn can block it again; if the user fixed the
        // environment, the goal can proceed. Failed steps remain scars and are
        // never silently retried by resume.
        if self.status == GoalStatus::Blocked
            && !self
                .sub_goals
                .iter()
                .any(|step| matches!(step.status, GoalStatus::Active | GoalStatus::Pending))
            && let Some((index, step)) = self
                .sub_goals
                .iter_mut()
                .enumerate()
                .find(|(_, step)| step.status == GoalStatus::Blocked)
        {
            let description = step.description.clone();
            step.status = GoalStatus::Active;
            self.status = GoalStatus::Active;
            self.push_event(
                "resume",
                format!("rechecking blocked step {}: {description}", index + 1),
            );
        }
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
            "  turns: {} spent · budget: {}\n",
            self.turns_spent,
            match self.turn_budget {
                Some(budget) => format!("{budget}"),
                None => "none — runs until done".to_string(),
            }
        ));
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
                GoalStatus::Blocked => "–",
            };
            out.push_str(&format!("    {:>2}. [{mark}] {}\n", i + 1, sg.description));
        }
        if end < total {
            out.push_str(&format!("    … {} later step(s)\n", total - end));
        }
        out.push_str(
            "  commands: /goal pause|resume|accept|status|edit …|limit …|budget …|team …|clear|export\n",
        );
        out
    }

    /// Markdown snapshot for human review (export-only; struct remains SoT).
    pub fn to_markdown(&self) -> String {
        let mut out = format!("# Goal\n\n**Objective:** {}\n\n", self.objective);
        let done = self.completed_count();
        let failed = self
            .sub_goals
            .iter()
            .filter(|s| s.status == GoalStatus::Failed)
            .count();
        out.push_str(&format!(
            "- status: {:?}\n- drive: {}\n- team: {}\n- progress: {done} done · {failed} failed · {} total\n",
            self.status,
            if self.is_paused() {
                self.pause_reason.as_str()
            } else {
                "running"
            },
            if self.team { "on" } else { "off" },
            self.sub_goals.len(),
        ));
        // The counter that distinguishes a run making progress from one walking
        // its checklist by abandoning every step. Only shown when non-zero so a
        // healthy plan stays uncluttered.
        if self.consecutive_skips > 0 {
            out.push_str(&format!(
                "- ⚠ steps abandoned in a row without a completion: {}\n",
                self.consecutive_skips
            ));
        }
        out.push_str("\n## Checklist\n\n");
        for (i, sg) in self.sub_goals.iter().enumerate() {
            let box_ = match sg.status {
                GoalStatus::Done => "[x]",
                GoalStatus::Active => "[>]",
                GoalStatus::Failed => "[!]",
                GoalStatus::Blocked => "[-]",
                GoalStatus::Pending => "[ ]",
            };
            out.push_str(&format!("{}. {} {}\n", i + 1, box_, sg.description));
            if sg.attempts > 0 {
                out.push_str(&format!("   - attempts: {}\n", sg.attempts));
            }
            if sg.unjudged_turns > 0 {
                out.push_str(&format!(
                    "   - unjudged turns (verification reached no verdict): {}\n",
                    sg.unjudged_turns
                ));
            }
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
            // Real progress clears the thrashing run.
            self.consecutive_skips = 0;
            // Let the invariant keeper select the next *drivable* step. A
            // recovered goal can contain a stale Done/Blocked/Failed entry
            // immediately after the cursor; blindly activating i + 1 would
            // resurrect that entry or leave the goal with an invalid cursor.
            self.rederive_status();
            if let Some(next) = self.active_sub_goal() {
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
                self.push_event(
                    "advance",
                    format!(
                        "completed step {}: {}; no drivable successor (goal {:?})",
                        i + 1,
                        done_desc,
                        self.status
                    ),
                );
            }
        }
    }

    /// Record an attempt on the active sub-goal (a verify failure the model
    /// couldn't fix, or a dead end). Returns `true` if the sub-goal still has
    /// retry budget (the agent should nudge "reconsider, don't repeat" and
    /// retry), `false` if an explicitly finite budget is exhausted and it is now
    /// `Failed` — in which case the overall goal is also `Failed` unless the
    /// agent chooses to skip. [`DEFAULT_SUBGOAL_RETRIES`] is an unlimited
    /// sentinel; a saturated diagnostic counter never exhausts it.
    pub fn record_failure(&mut self, note: impl Into<String>, max_retries: u32) -> bool {
        let Some(i) = self.active_index() else {
            return false;
        };
        let sg = &mut self.sub_goals[i];
        // A turn that produced a verdict clears the unjudged run, whichever way
        // the verdict went.
        sg.unjudged_turns = 0;
        sg.attempts = sg.attempts.saturating_add(1);
        push_clipped_note(sg, &note.into());
        if max_retries != u32::MAX && sg.attempts > max_retries {
            sg.status = GoalStatus::Failed;
            self.status = GoalStatus::Failed;
            false
        } else {
            true
        }
    }

    /// Record a turn that ended without verification reaching a verdict.
    ///
    /// Deliberately **not** [`record_failure`]: no attempt is charged and the
    /// step keeps its full retry budget, because nothing judged the work. The
    /// note still lands so the reason is visible in `/goal status` and the
    /// prompt. Returns `true` while the step may keep trying, `false` once
    /// [`MAX_UNJUDGED_TURNS`] consecutive turns have gone unjudged — at which
    /// point the caller should park the drive rather than keep spending turns
    /// on checks that never conclude.
    pub fn record_unjudged(&mut self, note: impl Into<String>) -> bool {
        let Some(i) = self.active_index() else {
            return false;
        };
        let sg = &mut self.sub_goals[i];
        sg.unjudged_turns = sg.unjudged_turns.saturating_add(1);
        push_clipped_note(sg, &note.into());
        sg.unjudged_turns < MAX_UNJUDGED_TURNS
    }

    /// Charge one drive turn against the budget. Returns `true` when that turn
    /// exhausted it (so the caller parks and reports). Always `false` when no
    /// budget is set.
    pub fn spend_turn(&mut self) -> bool {
        self.turns_spent = self.turns_spent.saturating_add(1);
        self.budget_exhausted()
    }

    /// Whether a set turn budget has been reached.
    pub fn budget_exhausted(&self) -> bool {
        self.turn_budget
            .is_some_and(|budget| self.turns_spent >= budget)
    }

    /// Turns left in the budget, if one is set.
    pub fn turns_remaining(&self) -> Option<u32> {
        self.turn_budget
            .map(|budget| budget.saturating_sub(self.turns_spent))
    }

    /// A human-readable account of where the goal actually got to.
    ///
    /// This is what a budgeted run is *for*: a goal that stops without saying
    /// what it finished, what it could not reach, and what is left is no more
    /// useful than one that ran forever.
    pub fn progress_report(&self) -> String {
        let done = self.completed_count();
        let failed = self
            .sub_goals
            .iter()
            .filter(|s| s.status == GoalStatus::Failed)
            .count();
        let blocked = self.blocked_steps().len();
        let remaining = self
            .sub_goals
            .iter()
            .filter(|s| matches!(s.status, GoalStatus::Pending | GoalStatus::Active))
            .count();
        let mut out = format!(
            "{done} done · {failed} failed · {blocked} blocked · {remaining} remaining (of {}), across {} turn(s)",
            self.sub_goals.len(),
            self.turns_spent,
        );
        if blocked > 0 {
            out.push_str("\nBlocked on prerequisites this environment doesn't have:");
            for (i, sub_goal) in self.blocked_steps() {
                let reason = sub_goal
                    .notes
                    .iter()
                    .find_map(|n| n.strip_prefix("blocked — missing prerequisite: "))
                    .unwrap_or("see notes");
                out.push_str(&format!("\n  {}. {reason}", i + 1));
            }
        }
        if let Some(next) = self
            .sub_goals
            .iter()
            .position(|s| matches!(s.status, GoalStatus::Pending | GoalStatus::Active))
        {
            out.push_str(&format!(
                "\nNext up: {}. {}",
                next + 1,
                self.sub_goals[next].description
            ));
        }
        out
    }

    /// Record a turn that landed verified work without finishing the active
    /// sub-goal. Returns `true` once the milestone has absorbed enough such
    /// turns ([`DECOMPOSE_AFTER_PRODUCTIVE_TURNS`]) to be worth splitting.
    pub fn record_productive_turn(&mut self) -> bool {
        let Some(i) = self.active_index() else {
            return false;
        };
        let sg = &mut self.sub_goals[i];
        sg.productive_turns = sg.productive_turns.saturating_add(1);
        sg.productive_turns >= DECOMPOSE_AFTER_PRODUCTIVE_TURNS
    }

    /// Set the active sub-goal aside as [`GoalStatus::Blocked`] and move to the
    /// next drivable step.
    ///
    /// Costs no retry budget — a missing prerequisite is not a failed attempt,
    /// and burning the budget on it would end with the step marked `Failed`,
    /// which tells the user their work was rejected rather than that their
    /// environment is short a dependency. It *does* count toward the skip run,
    /// so a plan whose every step is blocked parks rather than marching to the
    /// end reporting nothing but prerequisites.
    ///
    /// Returns whether drivable work remains.
    pub fn block_active(&mut self, prerequisite: impl Into<String>) -> bool {
        let Some(i) = self.active_index() else {
            return false;
        };
        let prerequisite = prerequisite.into();
        let description = self.sub_goals[i].description.clone();
        let sg = &mut self.sub_goals[i];
        sg.status = GoalStatus::Blocked;
        push_clipped_note(
            sg,
            &format!("blocked — missing prerequisite: {prerequisite}"),
        );
        self.consecutive_skips = self.consecutive_skips.saturating_add(1);
        self.push_event(
            "block",
            format!("step {}: {description} — {prerequisite}", i + 1),
        );
        self.rederive_status();
        self.status == GoalStatus::Active
    }

    /// Sub-goals set aside for a missing prerequisite, with their reasons — the
    /// actionable list a user can work through before resuming.
    pub fn blocked_steps(&self) -> Vec<(usize, &SubGoal)> {
        self.sub_goals
            .iter()
            .enumerate()
            .filter(|(_, s)| s.status == GoalStatus::Blocked)
            .collect()
    }

    /// How many sub-goals have actually completed.
    pub fn completed_count(&self) -> usize {
        self.sub_goals
            .iter()
            .filter(|s| s.status == GoalStatus::Done)
            .count()
    }

    /// Whether the drive is abandoning steps without ever completing one.
    ///
    /// This is the "thrashing looks like progress" guard: the cursor advancing
    /// past failed steps is indistinguishable from real movement in every
    /// surface the user sees, so a run where nothing has *ever* succeeded must
    /// stop and say so rather than walk the whole checklist.
    pub fn is_thrashing(&self) -> bool {
        self.completed_count() == 0 && self.consecutive_skips >= MAX_SKIPS_WITHOUT_COMPLETION
    }

    /// Skip the active sub-goal (mark `Failed` with a note) and advance to the
    /// next, so a blocked step doesn't halt the whole goal. The overall goal
    /// stays `Active` unless this was the last sub-goal.
    pub fn skip_active(&mut self, note: impl Into<String>) {
        self.skip_active_inner(note, false);
    }

    /// Stall-skip: same as [`Self::skip_active`], but the step is eligible for
    /// one second pass after the rest of the checklist lands.
    pub fn skip_stalled_active(&mut self, note: impl Into<String>) {
        self.skip_active_inner(note, true);
    }

    fn skip_active_inner(&mut self, note: impl Into<String>, stalled: bool) {
        if let Some(i) = self.active_index() {
            self.sub_goals[i].status = GoalStatus::Failed;
            push_clipped_note(&mut self.sub_goals[i], &note.into());
            if stalled {
                self.sub_goals[i].stall_skipped = true;
            }
            self.consecutive_skips = self.consecutive_skips.saturating_add(1);
            self.rederive_status();
        }
    }

    /// After the first sweep, return stall-skipped `Failed` steps to `Pending`
    /// once — only if something actually completed. Thrashing (zero completions)
    /// does not get a second pass.
    pub fn maybe_requeue_stall_skips(&mut self) -> usize {
        if self.completed_count() == 0 {
            return 0;
        }
        if self
            .sub_goals
            .iter()
            .any(|step| matches!(step.status, GoalStatus::Pending | GoalStatus::Active))
        {
            return 0;
        }
        let mut count = 0;
        for step in &mut self.sub_goals {
            if step.status == GoalStatus::Failed && step.stall_skipped && !step.requeued {
                step.status = GoalStatus::Pending;
                step.requeued = true;
                count += 1;
            }
        }
        if count > 0 {
            self.consecutive_skips = 0;
            self.rederive_status();
            self.push_event(
                "requeue",
                format!("{count} stall-skipped step(s) queued for a second pass"),
            );
        }
        count
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
        let mut known_descriptions = self
            .sub_goals
            .iter()
            .filter_map(|step| normalized_description(&step.description))
            .collect::<std::collections::HashSet<_>>();
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
                && let Some(normalized) = normalized_description(description)
                && known_descriptions.insert(normalized)
            {
                let mut sub_goal = SubGoal {
                    description: description.trim().to_string(),
                    status: GoalStatus::Pending,
                    attempts: 0,
                    notes: Vec::new(),
                    cap_continuations: 0,
                    barren_caps: 0,
                    split_depth: 0,
                    unjudged_turns: 0,
                    productive_turns: 0,
                    stall_skipped: false,
                    requeued: false,
                };
                if *status == GoalStatus::Done {
                    push_note_deduped(&mut sub_goal, CLAIM_NOTE);
                }
                self.sub_goals.push(sub_goal);
            }
        }
        self.rederive_status();
    }

    /// Retain rejected completion claims as durable context without changing
    /// status. The executor calls this after it has normalized unsupported
    /// implementation `done` claims back to their current state, so a later
    /// drive still knows to verify those claims rather than trusting them.
    pub(crate) fn record_unsupported_completion_claims(&mut self, indices: &[usize]) {
        for &index in indices {
            let Some(sub_goal) = self.sub_goals.get_mut(index) else {
                continue;
            };
            if sub_goal.status != GoalStatus::Done {
                push_note_deduped(sub_goal, CLAIM_NOTE);
            }
        }
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
        self.consecutive_skips = self.consecutive_skips.saturating_add(1);
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
        let mut known_descriptions = self
            .sub_goals
            .iter()
            .filter_map(|step| normalized_description(&step.description))
            .collect::<std::collections::HashSet<_>>();
        let mut appended = 0;
        for description in descriptions {
            if self
                .step_limit
                .is_some_and(|limit| self.sub_goals.len() >= limit)
            {
                break;
            }
            let Some(normalized) = normalized_description(description) else {
                continue;
            };
            if !known_descriptions.insert(normalized) {
                continue;
            }
            self.sub_goals.push(SubGoal {
                description: description.trim().to_string(),
                status: GoalStatus::Pending,
                attempts: 0,
                notes: Vec::new(),
                cap_continuations: 0,
                barren_caps: 0,
                split_depth: 0,
                unjudged_turns: 0,
                productive_turns: 0,
                stall_skipped: false,
                requeued: false,
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
                unjudged_turns: 0,
                productive_turns: 0,
                stall_skipped: false,
                requeued: false,
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
    pub(crate) fn rederive_status(&mut self) {
        if self.sub_goals.is_empty() {
            return;
        }
        if self.sub_goals.iter().all(|s| s.status == GoalStatus::Done) {
            self.status = GoalStatus::Done;
            return;
        }
        // Ensure exactly one drivable sub-goal is active (idempotent). Model
        // plan updates can claim multiple steps as active; keeping both would
        // make `active_index()` advance the first while the second remains an
        // orphaned active cursor on the next turn.
        // A blocked step is as undrivable as a failed one — no retry reaches it
        // — so it counts toward "nothing left to drive", but it is reported
        // separately because the user can actually clear it.
        let any_failed = self
            .sub_goals
            .iter()
            .any(|s| s.status == GoalStatus::Failed);
        let any_blocked = self
            .sub_goals
            .iter()
            .any(|s| s.status == GoalStatus::Blocked);
        let any_stuck = any_failed || any_blocked;
        let mut active_found = false;
        for sg in &mut self.sub_goals {
            match sg.status {
                GoalStatus::Active if !active_found => active_found = true,
                GoalStatus::Active => sg.status = GoalStatus::Pending,
                GoalStatus::Pending if !active_found => {
                    sg.status = GoalStatus::Active;
                    active_found = true;
                }
                _ => {}
            }
        }
        if any_stuck
            && !self
                .sub_goals
                .iter()
                .any(|s| s.status == GoalStatus::Active)
        {
            // A real failure dominates: claiming the goal is merely "blocked"
            // when something was judged and rejected would overstate how
            // recoverable it is. Only an all-blocked remainder reports Blocked,
            // which is the case where satisfying a prerequisite resumes work.
            self.status = if any_failed {
                GoalStatus::Failed
            } else {
                GoalStatus::Blocked
            };
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
        out.push_str(&format!(
            "Objective: {}\n",
            clip_chars(&self.objective, MAX_OBJECTIVE_CHARS)
        ));
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
                                clip_chars(&self.sub_goals[i].description, MAX_STEP_CHARS)
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
                GoalStatus::Blocked => '⛔',
                GoalStatus::Pending => '○',
            };
            out.push_str(&format!(
                "  {glyph} {}. {}\n",
                i + 1,
                clip_chars(&sg.description, MAX_STEP_CHARS)
            ));
            if sg.status == GoalStatus::Active && !sg.notes.is_empty() {
                out.push_str("     prior attempts (don't repeat these):\n");
                for n in sg.notes.iter().take(MAX_NOTES_IN_PROMPT) {
                    out.push_str(&format!("       — {}\n", clip_chars(n, MAX_NOTE_CHARS)));
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

/// Normalize plan descriptions for exact duplicate detection while keeping the
/// original text (minus surrounding whitespace) for display and prompts.
fn normalized_description(description: &str) -> Option<String> {
    let trimmed = description.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_ascii_lowercase())
}

/// Append `note` to the sub-goal unless an identical note is already recorded —
/// the model tends to resubmit the same plan several times per turn, and one
/// claim/regression note per step is signal; five are noise.
pub(crate) fn push_note_deduped(sub_goal: &mut SubGoal, note: &str) {
    push_clipped_note(sub_goal, note);
}

fn push_clipped_note(sub_goal: &mut SubGoal, note: &str) {
    let note = clip_chars(note, MAX_NOTE_CHARS);
    if note.is_empty() {
        return;
    }
    if !sub_goal.notes.iter().any(|n| n == &note) {
        sub_goal.notes.push(note);
        if sub_goal.notes.len() > MAX_NOTES_PER_STEP {
            sub_goal.notes.remove(0);
        }
    }
}

const MAX_OBJECTIVE_CHARS: usize = 500;
const MAX_STEP_CHARS: usize = 200;
pub(crate) const MAX_NOTE_CHARS: usize = 240;
const MAX_NOTES_PER_STEP: usize = 6;
pub(crate) const MAX_NOTES_IN_PROMPT: usize = 4;

pub(crate) fn clip_chars(text: &str, max: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{clipped}…")
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
    fn a_goal_saved_before_these_fields_existed_still_loads() {
        // Sessions on disk predate every counter added here. If any of them
        // failed to default, resuming an existing long-horizon goal would error
        // instead of picking up where it left off — losing exactly the progress
        // record these changes exist to protect.
        let legacy = r#"{
            "objective": "review plan.md and fully build this",
            "sub_goals": [
                {"description": "step one", "status": "Done", "attempts": 0, "notes": []},
                {"description": "step two", "status": "Active", "attempts": 2, "notes": ["a note"]}
            ],
            "status": "Active",
            "paused": true,
            "team": true
        }"#;
        let goal: Goal = serde_json::from_str(legacy).expect("legacy goal must deserialize");

        assert_eq!(goal.sub_goals.len(), 2);
        assert_eq!(goal.completed_count(), 1);
        assert!(goal.is_paused());
        // Every field added across these changes defaults sanely.
        assert_eq!(goal.consecutive_skips, 0);
        assert_eq!(goal.turn_budget, None);
        assert_eq!(goal.turns_spent, 0);
        assert_eq!(goal.sub_goals[1].unjudged_turns, 0);
        assert_eq!(goal.sub_goals[1].productive_turns, 0);
        assert_eq!(goal.sub_goals[1].cap_continuations, 0);
        // And the pre-existing state survives untouched.
        assert_eq!(goal.sub_goals[1].attempts, 2);
        assert!(!goal.is_thrashing(), "a legacy goal must not read as stuck");
    }

    #[test]
    fn fresh_goals_have_no_default_turn_budget() {
        let g = goal();
        assert_eq!(g.turn_budget, None);
        assert!(!g.budget_auto);
        assert!(!g.budget_exhausted());
    }

    #[test]
    fn plan_growth_does_not_install_a_default_turn_budget() {
        let mut g = Goal::new("ship it", vec!["one".into()]);
        let grown: Vec<String> = (0..80).map(|i| format!("step {i}")).collect();
        g.append_missing(&grown);
        assert_eq!(g.turn_budget, None);
        assert!(!g.budget_auto);
    }

    #[test]
    fn a_legacy_automatic_budget_is_removed_and_reopened() {
        let mut g = Goal::new("ship it", vec!["one".into()]);
        g.turn_budget = Some(25);
        g.budget_auto = true;
        g.pause(GoalPauseReason::Budget);

        assert!(g.clear_legacy_automatic_budget());
        assert_eq!(g.turn_budget, None);
        assert!(!g.budget_auto);
        assert!(!g.is_paused());
        assert_eq!(g.pause_reason, GoalPauseReason::None);
        assert!(!g.clear_legacy_automatic_budget());
    }

    #[test]
    fn an_explicit_budget_stops_the_rescaling() {
        let mut g = Goal::new("ship it", vec!["one".into()]);
        g.turn_budget = Some(7);
        g.budget_auto = false; // as `/goal budget 7` does
        let grown: Vec<String> = (0..40).map(|i| format!("step {i}")).collect();
        g.append_missing(&grown);
        assert_eq!(
            g.turn_budget,
            Some(7),
            "a number the user chose must not move under them"
        );
    }

    #[test]
    fn compatibility_auto_budget_helper_is_unlimited() {
        assert_eq!(auto_budget_for(0), u32::MAX);
        assert_eq!(auto_budget_for(1), u32::MAX);
        assert_eq!(auto_budget_for(100_000), u32::MAX);
    }

    #[test]
    fn a_turn_budget_bounds_an_open_ended_objective() {
        // "fully build this" against a multi-phase plan has no reachable end
        // state, so without a ceiling it simply runs until someone notices.
        let mut g = goal();
        // `/goal budget off` — the explicit opt-out.
        g.turn_budget = None;
        g.budget_auto = false;
        assert!(!g.budget_exhausted(), "no budget set = runs until done");
        assert_eq!(g.turns_remaining(), None);
        assert!(!g.spend_turn(), "spending against no budget never exhausts");

        g.turn_budget = Some(2);
        g.turns_spent = 0;
        assert!(!g.spend_turn(), "one of two");
        assert_eq!(g.turns_remaining(), Some(1));
        assert!(g.spend_turn(), "the second turn exhausts it");
        assert!(g.budget_exhausted());
        assert_eq!(g.turns_remaining(), Some(0));
    }

    #[test]
    fn the_progress_report_accounts_for_every_step() {
        // A goal that stops without saying what it finished, what it couldn't
        // reach, and what's left is no more useful than one that ran forever.
        let mut g = Goal::new(
            "ship it",
            vec!["one".into(), "two".into(), "three".into(), "four".into()],
        );
        g.advance(); // one: done
        g.block_active("a running PostgreSQL"); // two: blocked
        g.record_failure("verification failed", 0); // three: failed
        g.turns_spent = 7;

        let report = g.progress_report();
        assert!(report.contains("1 done"), "{report}");
        assert!(report.contains("1 failed"), "{report}");
        assert!(report.contains("1 blocked"), "{report}");
        assert!(report.contains("across 7 turn(s)"), "{report}");
        assert!(
            report.contains("a running PostgreSQL"),
            "the actionable prerequisite must appear: {report}"
        );
        assert!(
            report.contains("Next up: 4."),
            "the user needs to know where it would resume: {report}"
        );
    }

    #[test]
    fn blocking_a_step_costs_no_retry_budget_and_is_not_a_failure() {
        // A missing prerequisite is not a rejected attempt. Marking it `Failed`
        // tells the user their work was judged and found wanting, and hides the
        // one thing they can act on.
        let mut g = goal();
        assert!(g.block_active("a running PostgreSQL reachable via DATABASE_URL"));

        assert_eq!(g.sub_goals[0].status, GoalStatus::Blocked);
        assert_ne!(g.sub_goals[0].status, GoalStatus::Failed);
        assert_eq!(g.sub_goals[0].attempts, 0, "no retry budget spent");
        assert_eq!(g.active_index(), Some(1), "the drive moves on");
        assert_eq!(g.status, GoalStatus::Active);

        let blocked = g.blocked_steps();
        assert_eq!(blocked.len(), 1);
        assert!(
            blocked[0].1.notes.iter().any(|n| n.contains("PostgreSQL")),
            "the prerequisite must be recorded verbatim: {:?}",
            blocked[0].1.notes
        );
    }

    #[test]
    fn a_wholly_blocked_plan_reports_blocked_not_failed() {
        let mut g = goal();
        g.block_active("no database");
        g.block_active("no database");
        assert!(!g.block_active("no database"), "nothing left to drive");
        assert_eq!(
            g.status,
            GoalStatus::Blocked,
            "the goal is waiting on prerequisites, not broken"
        );

        // A genuine failure alongside blocks dominates — claiming merely
        // "blocked" would overstate how recoverable the run is.
        let mut mixed = goal();
        mixed.block_active("no database");
        mixed.record_failure("verification failed", 0);
        mixed.block_active("no tofu");
        assert_eq!(mixed.status, GoalStatus::Failed);
    }

    #[test]
    fn resuming_a_wholly_blocked_plan_rechecks_the_first_prerequisite() {
        let mut g = goal();
        g.block_active("database");
        g.block_active("database");
        g.block_active("database");
        assert_eq!(g.status, GoalStatus::Blocked);
        assert_eq!(g.active_index(), None);

        g.resume();

        assert_eq!(g.status, GoalStatus::Active);
        assert_eq!(g.active_index(), Some(0));
        assert_eq!(g.sub_goals[0].status, GoalStatus::Active);
        assert_eq!(g.sub_goals[1].status, GoalStatus::Blocked);
        assert!(g.should_auto_drive());
        assert!(
            g.events
                .iter()
                .any(|event| event.detail.contains("rechecking blocked step 1"))
        );
    }

    #[test]
    fn an_oversized_milestone_is_flagged_after_repeated_productive_turns() {
        // A model that ends its turns cleanly never trips the step cap, so
        // `cap_continuations` stays zero while a days-long milestone absorbs
        // turn after turn of real work. Productive-but-unfinished turns are the
        // signal that it should be split.
        let mut g = goal();
        for turn in 1..DECOMPOSE_AFTER_PRODUCTIVE_TURNS {
            assert!(
                !g.record_productive_turn(),
                "turn {turn} is not yet evidence of an oversized step"
            );
        }
        assert!(
            g.record_productive_turn(),
            "past the threshold the milestone is too big for one step"
        );
        assert_eq!(g.sub_goals[0].cap_continuations, 0, "no step cap involved");
        assert_eq!(
            g.sub_goals[0].attempts, 0,
            "productive turns aren't failures"
        );
    }

    #[test]
    fn unjudged_turns_do_not_spend_the_retry_budget() {
        // The incident: verification timed out, and each timeout was charged as
        // a failed attempt until the step was marked Failed and skipped — for a
        // defect in the checks, not the work.
        let mut g = goal();
        assert!(g.record_unjudged("verification reached no verdict"));
        assert_eq!(g.sub_goals[0].attempts, 0, "no attempt may be charged");
        assert_eq!(g.sub_goals[0].status, GoalStatus::Active);
        assert_eq!(g.sub_goals[0].unjudged_turns, 1);

        // Repeating the same note must not spam the prompt.
        assert!(!g.record_unjudged("verification reached no verdict"));
        assert_eq!(g.sub_goals[0].notes.len(), 1);
        assert_eq!(
            g.sub_goals[0].unjudged_turns, MAX_UNJUDGED_TURNS,
            "past the cap the caller is told to park"
        );
        assert_eq!(
            g.sub_goals[0].attempts, 0,
            "still nothing charged to the budget"
        );
        assert_eq!(g.status, GoalStatus::Active, "goal must not be failed");

        // A turn that does reach a verdict clears the unjudged run.
        g.record_failure("verification failed", DEFAULT_SUBGOAL_RETRIES);
        assert_eq!(g.sub_goals[0].unjudged_turns, 0);
        assert_eq!(g.sub_goals[0].attempts, 1);
    }

    #[test]
    fn skipping_every_step_without_a_completion_is_thrashing() {
        // A cursor that advances only by abandoning steps looks exactly like
        // progress. After two skips with nothing ever completed, the drive must
        // report itself as stuck rather than walk the rest of the plan.
        let mut g = goal();
        assert!(!g.is_thrashing(), "a fresh goal is not thrashing");

        g.skip_active("blocked");
        assert_eq!(g.consecutive_skips, 1);
        assert!(!g.is_thrashing(), "one skip is bad luck, not a pattern");

        g.skip_active("blocked again");
        assert_eq!(g.consecutive_skips, MAX_SKIPS_WITHOUT_COMPLETION);
        assert_eq!(g.completed_count(), 0);
        assert!(g.is_thrashing(), "two skips, nothing ever done");
    }

    #[test]
    fn a_single_completion_clears_thrashing() {
        // Skips after real progress are the case `continue_past_failure` exists
        // for — one stuck milestone must not park a run that is working.
        let mut g = goal();
        g.advance(); // step one genuinely completes
        assert_eq!(g.completed_count(), 1);
        g.skip_active("step two blocked");
        g.skip_active("step three blocked");
        assert!(
            !g.is_thrashing(),
            "a goal that has completed work is not thrashing, however many skips follow"
        );
    }

    #[test]
    fn completion_resets_the_skip_run() {
        let mut g = goal();
        g.skip_active("blocked");
        assert_eq!(g.consecutive_skips, 1);
        g.advance();
        assert_eq!(g.consecutive_skips, 0, "real progress clears the run");
    }

    #[test]
    fn markdown_export_shows_progress_and_attempts() {
        // The export is the file people open to check on a long run; it has to
        // show whether steps are completing or merely being abandoned.
        let mut g = goal();
        g.record_failure("verification failed", DEFAULT_SUBGOAL_RETRIES);
        g.record_unjudged("verification reached no verdict");
        let md = g.to_markdown();
        assert!(md.contains("progress: 0 done · 0 failed · 3 total"), "{md}");
        assert!(md.contains("attempts: 1"), "{md}");
        assert!(md.contains("unjudged turns"), "{md}");

        let mut skipped = goal();
        skipped.skip_active("blocked");
        assert!(
            skipped
                .to_markdown()
                .contains("steps abandoned in a row without a completion: 1"),
            "{}",
            skipped.to_markdown()
        );
    }

    #[test]
    fn new_goal_activates_first_sub_goal() {
        let g = goal();
        assert_eq!(g.status, GoalStatus::Active);
        assert_eq!(g.active_index(), Some(0));
        assert_eq!(g.sub_goals[1].status, GoalStatus::Pending);
    }

    #[test]
    fn goal_ignores_blank_plan_entries() {
        let g = Goal::new(
            "build it",
            vec![
                "  ".into(),
                "first real step".into(),
                " FIRST REAL STEP ".into(),
                "".into(),
            ],
        );
        assert_eq!(g.sub_goals.len(), 1);
        assert_eq!(g.sub_goals[0].description, "first real step");
        assert_eq!(g.active_index(), Some(0));
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
    fn advance_skips_non_drivable_entries_in_a_recovered_goal() {
        let mut g = goal();
        g.sub_goals[1].status = GoalStatus::Done;

        g.advance();

        assert_eq!(g.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(g.sub_goals[1].status, GoalStatus::Done);
        assert_eq!(g.active_index(), Some(2));
        assert_eq!(g.status, GoalStatus::Active);
    }

    #[test]
    fn advance_preserves_terminal_status_when_recovered_steps_are_stuck() {
        let mut blocked = goal();
        blocked.sub_goals[1].status = GoalStatus::Blocked;
        blocked.sub_goals[2].status = GoalStatus::Failed;

        blocked.advance();

        assert_eq!(blocked.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(blocked.active_index(), None);
        assert_eq!(
            blocked.status,
            GoalStatus::Failed,
            "failure remains terminal"
        );

        let mut only_blocked = goal();
        only_blocked.sub_goals[1].status = GoalStatus::Blocked;
        only_blocked.sub_goals[2].status = GoalStatus::Done;

        only_blocked.advance();

        assert_eq!(only_blocked.active_index(), None);
        assert_eq!(only_blocked.status, GoalStatus::Blocked);
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
    fn default_failure_retries_are_unlimited_even_after_counter_saturation() {
        let mut g = goal();
        for attempt in 1..=6 {
            assert!(
                g.record_failure(
                    format!("productive retry {attempt}"),
                    DEFAULT_SUBGOAL_RETRIES,
                ),
                "the ordinary retry policy must remain active past the old cap"
            );
        }
        assert_eq!(g.sub_goals[0].attempts, 6);
        assert_eq!(g.sub_goals[0].status, GoalStatus::Active);
        assert_eq!(g.status, GoalStatus::Active);

        g.sub_goals[0].attempts = u32::MAX;
        assert!(
            g.record_failure("saturated diagnostic counter", DEFAULT_SUBGOAL_RETRIES),
            "u32::MAX is an unlimited sentinel, not a reachable cap"
        );
        assert_eq!(g.sub_goals[0].attempts, u32::MAX);
        assert_eq!(g.sub_goals[0].status, GoalStatus::Active);
        assert_eq!(g.status, GoalStatus::Active);
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
    fn skip_active_does_not_reactivate_a_done_successor() {
        let mut g = goal();
        g.sub_goals[1].status = GoalStatus::Done;

        g.skip_active("blocked on upstream API");

        assert_eq!(g.sub_goals[0].status, GoalStatus::Failed);
        assert_eq!(g.sub_goals[1].status, GoalStatus::Done);
        assert_eq!(g.active_index(), Some(2));
        assert_eq!(g.status, GoalStatus::Active);
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
    fn apply_plan_statuses_normalizes_multiple_active_claims() {
        let mut g = goal();
        g.apply_plan_statuses(&["pending", "active", "active"]);

        assert_eq!(g.active_index(), Some(0));
        assert_eq!(
            g.sub_goals
                .iter()
                .filter(|step| step.status == GoalStatus::Active)
                .count(),
            1,
            "a plan update must leave one active cursor"
        );
        assert_eq!(g.sub_goals[1].status, GoalStatus::Pending);
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
    fn prompt_section_clips_huge_objective_and_notes() {
        let mut g = Goal::new("O".repeat(2_000), vec!["S".repeat(800)]);
        g.record_failure("N".repeat(1_000), 3);
        let section = g.prompt_section().expect("renders");
        assert!(
            section.chars().count() < 3_000,
            "goal prompt must stay bounded: {}",
            section.chars().count()
        );
        assert!(!section.contains(&"O".repeat(600)), "{section}");
        assert!(!section.contains(&"N".repeat(400)), "{section}");
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
    fn apply_plan_dedupes_and_skips_blank_discoveries() {
        let mut g = Goal::new("small", vec!["first".into()]);
        let anchor = g.active_index();
        g.apply_plan(
            &[
                ("first".into(), GoalStatus::Active),
                ("second".into(), GoalStatus::Pending),
                ("  ".into(), GoalStatus::Pending),
                ("SECOND".into(), GoalStatus::Done),
            ],
            anchor,
        );
        assert_eq!(g.sub_goals.len(), 2);
        assert_eq!(g.sub_goals[1].description, "second");
        assert_eq!(g.sub_goals[1].status, GoalStatus::Pending);
        assert!(
            g.sub_goals[1].notes.is_empty(),
            "a duplicate late claim must not mutate the already-appended step"
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
    fn append_missing_skips_blank_discoveries() {
        let mut g = Goal::new("small", vec!["first".into()]);
        g.advance();
        assert_eq!(
            g.append_missing(&[" ".into(), "".into(), "second".into()]),
            1
        );
        assert_eq!(g.sub_goals[1].description, "second");
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
    fn drive_state_ignores_accounting_and_retry_diagnostics_but_detects_advancement() {
        let before = goal();
        let mut spent = before.clone();
        spent.spend_turn();
        assert!(
            !spent.drive_state_changed_since(&before),
            "spending a synthetic drive turn is not progress"
        );

        let mut retried = spent.clone();
        retried.record_failure("retry this step", DEFAULT_SUBGOAL_RETRIES);
        assert!(
            !retried.drive_state_changed_since(&before),
            "retry bookkeeping must not disguise a no-progress loop"
        );

        let mut advanced = retried;
        advanced.advance();
        assert!(
            advanced.drive_state_changed_since(&before),
            "advancing the active step is structural progress"
        );
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
