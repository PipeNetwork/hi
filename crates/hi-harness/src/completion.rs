//! Harness-owned turn completion policy.
//!
//! The turn loop asks the model for the next action; this module decides
//! whether that response may finish the turn. Hints are the voice of a
//! demand, not a second stop machine. Progress is facts: mutation, verify,
//! an open plan, a user-visible answer, consecutive stall rounds.

use hi_ai::{Content, Message};
use hi_tools::{PlanStatus, PlanStep, is_coordination, is_inspect_tool, plan_steps_from_arguments};

/// Consecutive stall-only tool rounds before the harness demands Edit / Verify.
///
/// The budget only applies while the turn owes concrete work (an edit on
/// Implement or an open plan, tests on Fix). When nothing is owed the stall
/// demand would be a bare Verdict, and there opening new files *is* the
/// work: a review of a 1000-file tree is many read rounds, not a stall.
/// Those turns proceed until identical repeats hit
/// [`Event::ToolsFinished::inspect_repeat_capped`].
pub const STALL_ROUNDS_BEFORE_DEMAND: u32 = 2;
/// Keep re-hinting the stall demand until this many inspect-only rounds.
/// Identical inspect-repeat / probe refusals still error after one extra hint.
pub const STALL_ROUNDS_BEFORE_ERROR: u32 = 8;
/// One distinct stall-demand kind, then Error if the model still stalls on a
/// ModelStop cop-out. Tool rounds keep the hint until [`STALL_ROUNDS_BEFORE_ERROR`].
pub const DEMAND_CONTINUATIONS: u32 = 1;
/// Empty / reasoning-only stops after tools get two VisibleAnswer hints, then Error.
pub const EMPTY_AFTER_TOOLS_CONTINUATIONS: u32 = 2;

pub const EMPTY_AFTER_TOOLS_HINT: &str = "\
Your last response ended with no user-visible answer after tool work. \
Do not repeat the same grep/read/list/write. If a tool result already \
answers the user, write that reply now. If a search returned no matches \
and the user asked you to add the missing code, add it with edit/write. \
Do not change a file just to restate text that is already there. Do not \
stop without a user-visible reply.";

pub const UNVERIFIED_FIX_HINT: &str = "\
You were asked to review and fix. Run `cargo test` (or the project's test \
command) now to verify the current tree. If tests fail, fix them with `edit`. \
If they pass, give a short verdict and stop. Do not keep grepping.";

#[allow(dead_code)]
pub const NEXT_ACTION_EDIT_HINT: &str = "\
You were asked to review and fix. Stop inspecting and call `edit`/`write` on \
the failing code, or run `cargo test` if you do not yet know what fails.";

/// Edit demand when a test run cannot close the active step (a lone
/// untouched step, or this turn's test-run close is spent), so it does not
/// offer `update_plan`.
pub const EDIT_NOW_HINT: &str = "\
Stop inspecting. Your next action MUST be `edit` or `write` on the files that \
need to change; if a plan is open, implement its active step. Do not start a \
new grep or list; if you were paging a file, you have enough. Never add \
filler comments or no-op edits to satisfy this.";

/// Edit demand while a test run can still close the active step
/// ([`TurnFacts::test_close_available`]). It must name that route: an
/// edit-only demand there leaves the model padding a finished file with
/// comments until plan_stall.
pub const PLAN_EXECUTE_HINT: &str = "\
Stop inspecting. Implement the active plan step now: your next action MUST \
be `edit` or `write` on the files that need to change. Do not start a new \
grep or list; if you were paging a file, you have enough. Only if the active \
step is already fully implemented in the current tree: run the project's \
tests, then call update_plan to mark it done and set the next step active. \
Never add filler comments or no-op edits to satisfy this.";

/// Edit demand after a step closed this turn without edits
/// ([`TurnFacts::closed_step_without_edits`]) and a test run can no longer
/// close another. Stopping is allowed: the next turn gets its own.
pub const NEXT_STEP_HINT: &str = "\
A plan step already closed this turn. If the active step still needs changes, \
make them now with `edit` or `write`; do not start a new grep or list. If it \
is already implemented, stop and say so: the next turn can close it after a \
passing test run. Never add filler comments or no-op edits.";

/// Edit demand after the tests failed on the active step's edit.
pub const FAILING_TESTS_HINT: &str = "\
The tests failed after your last edit. Read that failure output and fix it \
with `edit`/`write` now; the step cannot close until the tests pass. Do not \
re-run them before changing a file, and never add filler edits.";

/// Sent once the active step was edited and the tests passed, but the model
/// stopped without `update_plan`. An edit demand here instead pushes the model
/// into filler edits (repeated comment lines) just to keep the turn alive.
pub const PLAN_CLOSE_HINT: &str = "\
Your edits are applied and the tests passed, but the active plan step is \
still open. If that step is complete, call update_plan now: mark it done, set \
the next unfinished step active, then start that step with edit/write. If \
part of the step is still missing, edit it. Do not recap, and never add \
filler comments or no-op edits to keep the turn going.";

pub const FAILED_EDIT_HINT: &str = "\
Your last edit/write did not apply. Do not retry the same old_string and do \
not run cargo test again. Read the target file (or the closest-matching \
region from the error) and copy old_string from the current text, then edit. \
If the new_string is already in the file, call update_plan: mark the active \
step done and set the next unfinished step active. Do not recap status.";

pub const INSPECT_STOP_VERDICT_HINT: &str = "\
Stop inspecting. Write a short user-visible answer now: what to change, or \
that you will not change anything. Do not grep or read the same files again.";

const EMPTY_STOP_MSG: &str = "model stopped after tool work with no user-visible answer";
const INSPECT_STOP_MSG: &str = "stopped inspecting without applying the requested changes";
const UNVERIFIED_STOP_MSG: &str =
    "changes were applied but the project's tests never ran after the last edit";
const PLAN_STALL_MSG: &str = "plan still has unfinished steps but the turn stopped without an edit";
const PLAN_STALL_FAILING_MSG: &str =
    "plan still has unfinished steps and the tests failed after the last edit";
const PLAN_KEPT_OPEN_MSG: &str = "the model kept the active plan step open after the tests passed, then stopped without another edit";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    Review,
    Fix,
    Implement,
}

impl Intent {
    pub fn from_prompt(text: &str) -> Self {
        if user_asked_to_implement(text) {
            Self::Implement
        } else if user_asked_to_fix(text) {
            Self::Fix
        } else {
            Self::Review
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Demand {
    Edit,
    Verify,
    Verdict,
    VisibleAnswer,
    /// The active step was edited and tested green; ask for `update_plan`.
    ClosePlanStep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// Model returned no tool calls.
    ModelStop,
    /// A tool batch finished. `inspect_repeat_capped` is true after two
    /// inspect-repeat or detached-probe refusals this turn.
    ToolsFinished { inspect_repeat_capped: bool },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Ask the model again with a one-shot (unpersisted) hint.
    Continue {
        demand: Demand,
        hint: &'static str,
    },
    /// Call the model again with no extra hint (normal tool loop).
    Proceed,
    Complete,
    /// The [`Demand::ClosePlanStep`] hint went unanswered. The harness marks
    /// the active step done itself and completes the turn instead of ending
    /// in `plan_stall` with the work finished.
    ClosePlanStep,
    Error {
        kind: &'static str,
        message: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnFacts {
    pub intent: Intent,
    pub used_tools: bool,
    pub visible_answer: bool,
    pub mutated: bool,
    pub ran_verify: bool,
    pub plan_open: bool,
    pub consecutive_stall: u32,
    pub empty_after_tools: u32,
    pub demand_continuations: u32,
    pub last_demand: Option<Demand>,
    /// Tool results from the round after a stall demand already went out.
    /// The model sees those results once; the next stall is Error.
    pub last_chance: bool,
    /// Unspent mutations (plus at most one post-edit verify) that may mark
    /// plan steps `done`. A bulk `update_plan` close cannot spend more than
    /// this, so one edit cannot retire an eight-step checklist.
    pub work_credits: u32,
    /// `cargo test` after a mutation grants one credit, not one per re-run.
    pub verify_credit_granted: bool,
    /// The first `update_plan`-only round posted the checklist; it is not a
    /// stall. Later coordination-only rounds still increment.
    pub opening_plan_grace: bool,
    /// Workspace paths already `read` this turn. A later `read` of the same
    /// path with a new offset is pagination, not a unique-file stall.
    pub seen_read_paths: Vec<String>,
    /// Consecutive same-path pagination rounds. Caps paging one huge file
    /// forever without blocking unique-file exploration.
    pub pagination_rounds: u32,
    /// An edit/write/patch was attempted and failed this turn.
    pub failed_mutation: bool,
    /// One `update_plan` credit after a failed edit plus a read, so an
    /// already-implemented step can close without a fake patch.
    pub reconcile_credit_granted: bool,
    /// An edit landed since the active plan step opened, this turn or an
    /// earlier one (see [`crate::PlanDrive::step_edited`]).
    pub step_work: bool,
    /// The last verify run failed. A red run proves nothing about the step,
    /// so it grants no reconcile credit and cannot close the step.
    pub verify_failed: bool,
    /// Steps in the current plan. Until it is edited, a lone step cannot be
    /// closed by a test run alone.
    pub plan_steps: usize,
    /// A plan step closed this turn. Like a reconcile, that satisfies an
    /// Implement turn that made no edit because the code was already there.
    pub closed_step: bool,
    /// After passing tests the model posted `update_plan` with the active
    /// step still open: it says the step is not finished, so the harness
    /// must not close it for the model until another edit lands.
    pub step_kept_open: bool,
}

impl TurnFacts {
    pub fn new(intent: Intent, plan_open: bool) -> Self {
        Self {
            intent,
            used_tools: false,
            visible_answer: false,
            mutated: false,
            ran_verify: false,
            plan_open,
            consecutive_stall: 0,
            empty_after_tools: 0,
            demand_continuations: 0,
            last_demand: None,
            last_chance: false,
            work_credits: 0,
            verify_credit_granted: false,
            opening_plan_grace: false,
            seen_read_paths: Vec::new(),
            pagination_rounds: 0,
            failed_mutation: false,
            reconcile_credit_granted: false,
            step_work: false,
            verify_failed: false,
            plan_steps: 0,
            closed_step: false,
            step_kept_open: false,
        }
    }

    pub fn grant_mutation_credit(&mut self) {
        self.work_credits = self.work_credits.saturating_add(1);
    }

    /// A missed edit plus a later read (or an "already in the file" miss)
    /// may close exactly one plan step.
    pub fn grant_reconcile_credit(&mut self) {
        if self.failed_mutation && !self.reconcile_credit_granted {
            self.work_credits = self.work_credits.saturating_add(1);
            self.reconcile_credit_granted = true;
        }
    }

    /// A passing test run alone may close one step per turn: a step finished
    /// in an earlier turn (plan drive runs a turn per step), one an earlier
    /// step's edits already covered, or one that needs no edit ("Baseline:
    /// run cargo test"). See [`tests_may_close_plan_step`].
    pub fn tests_may_close_step(&self) -> bool {
        tests_may_close(self.plan_open, self.step_work, self.plan_steps)
    }

    /// Running the tests and calling `update_plan` can still close the
    /// active step this turn, so a hint may offer that route.
    pub fn test_close_available(&self) -> bool {
        self.tests_may_close_step() && (!self.verify_credit_granted || self.work_credits > 0)
    }

    /// A step closed this turn on tests alone ("Baseline: run cargo test",
    /// or code an earlier turn wrote) and nothing was edited. If the next
    /// step is already done too, only a fresh turn can close it, so stopping
    /// here ends the turn; plan drive opens that step next. A turn with
    /// edits keeps its post-edit test credit for that and is still pushed to
    /// finish, so "Remaining: …, not implementing those" stays a plan_stall.
    pub fn closed_step_without_edits(&self) -> bool {
        self.closed_step && !self.mutated
    }

    /// One credit per turn for tests after an edit, or for a passing run
    /// whenever [`Self::tests_may_close_step`]. Without the second case a
    /// finished step could only close through yet another (filler) edit.
    pub fn grant_verify_credit(&mut self) {
        if self.verify_credit_granted {
            return;
        }
        if self.mutated || (!self.verify_failed && self.tests_may_close_step()) {
            self.work_credits = self.work_credits.saturating_add(1);
            self.verify_credit_granted = true;
        }
    }

    /// Judge the test-run credit against a proposed `update_plan` before
    /// [`accept_plan`] spends credits, so tests that already passed this turn
    /// also count for a plan posted after them.
    pub fn preview_plan(&mut self, previous: &[PlanStep], proposed: &[PlanStep]) {
        self.plan_open = proposed.iter().any(|step| !was_done(previous, step));
        self.plan_steps = proposed.len();
        if self.ran_verify {
            self.grant_verify_credit();
        }
    }

    /// The active step was edited, the tests after that edit passed, and a
    /// credit is left to close it. Nothing is owed but `update_plan`.
    pub fn ready_to_close_step(&self) -> bool {
        self.plan_open
            && matches!(self.intent, Intent::Implement | Intent::Fix)
            && self.step_work
            && self.ran_verify
            && !self.verify_failed
            && self.work_credits > 0
            && !self.step_kept_open
    }

    /// Live ~/drme: tests passed, the model answered the close demand with
    /// `update_plan` keeping "Verify dav1d + ffmpeg + tests" active because
    /// dav1d still rejected the stream, and the harness closed the step
    /// anyway. The plan then read 4/4, was dismissed, and nothing resumed.
    /// A plan posted before the tests ran is not such an answer.
    pub fn note_plan_update(&mut self, previous: &[PlanStep], proposed: &[PlanStep]) {
        self.step_kept_open =
            self.ran_verify && !self.verify_failed && keeps_active_step_open(previous, proposed);
    }

    /// Remaining plan steps on Implement and Fix are still an edit demand
    /// after a partial patch. Verify wins if tests have not run since the
    /// last edit.
    pub fn wants_edit(&self) -> bool {
        if self.wants_verify() {
            return false;
        }
        match self.intent {
            Intent::Implement => {
                self.plan_open
                    || (!self.mutated && !self.reconcile_credit_granted && !self.closed_step)
            }
            Intent::Fix => self.plan_open,
            Intent::Review => false,
        }
    }

    pub fn wants_verify(&self) -> bool {
        if self.ran_verify {
            return false;
        }
        match self.intent {
            Intent::Fix => true,
            Intent::Implement => self.mutated,
            Intent::Review => false,
        }
    }

    pub fn note_continue(&mut self, demand: Demand) {
        match demand {
            Demand::VisibleAnswer => {
                self.empty_after_tools = self.empty_after_tools.saturating_add(1);
            }
            other => {
                if self.last_demand == Some(other) {
                    self.demand_continuations = self.demand_continuations.saturating_add(1);
                } else {
                    self.last_demand = Some(other);
                    self.demand_continuations = 1;
                }
            }
        }
    }

    /// Update consecutive stall after a tool round.
    ///
    /// Mutation (or a round that closed a plan step) always resets.
    /// Stall-only rounds increment. `cargo test`
    /// while a plan is still open and nothing was edited does **not** reset
    /// — that is the shipping stall (tests, then eighty reads). The first
    /// `update_plan`-only round is checklist progress, not a stall.
    #[cfg(test)]
    pub fn note_round(
        &mut self,
        mutated_this_round: bool,
        stall_only: bool,
        verify_this_round: bool,
    ) {
        self.note_round_kind(
            mutated_this_round,
            stall_only,
            verify_this_round,
            false,
            false,
        );
    }

    pub fn note_round_kind(
        &mut self,
        mutated_this_round: bool,
        stall_only: bool,
        verify_this_round: bool,
        coordination_only: bool,
        pagination_only: bool,
    ) {
        if mutated_this_round {
            self.consecutive_stall = 0;
            self.last_demand = None;
            self.demand_continuations = 0;
            self.last_chance = false;
            self.pagination_rounds = 0;
            self.seen_read_paths.clear();
            return;
        }
        if stall_only {
            if coordination_only && !self.opening_plan_grace {
                self.opening_plan_grace = true;
                return;
            }
            if pagination_only {
                self.pagination_rounds = self.pagination_rounds.saturating_add(1);
                return;
            }
            // A newly opened path is unique-file progress; restart the
            // per-file page cap so protocol.rs pages do not spend server.rs
            // slots.
            self.consecutive_stall = self.consecutive_stall.saturating_add(1);
            self.pagination_rounds = 0;
            return;
        }
        if verify_this_round && self.plan_open && !self.mutated {
            return;
        }
        self.consecutive_stall = 0;
        self.pagination_rounds = 0;
    }

    /// How close this turn is to the inspect cap. Unique-file rounds and
    /// same-path pagination each have their own counter; the cap uses the
    /// higher one so paging `server.rs` does not spend unique-file slots.
    pub fn stall_pressure(&self) -> u32 {
        self.consecutive_stall.max(self.pagination_rounds)
    }

    pub fn observe_read_paths(&mut self, calls: &[crate::pipe::ToolCall]) {
        for call in calls {
            if let Some(path) = read_tool_path(&call.name, &call.arguments)
                && !self.seen_read_paths.iter().any(|seen| seen == &path)
            {
                self.seen_read_paths.push(path);
            }
        }
    }
}

pub fn decide(event: Event, facts: &mut TurnFacts) -> Decision {
    match event {
        Event::ModelStop => decide_model_stop(facts),
        Event::ToolsFinished {
            inspect_repeat_capped,
        } => decide_tools_finished(facts, inspect_repeat_capped),
    }
}

fn decide_model_stop(facts: &mut TurnFacts) -> Decision {
    // Review/Q&A and a Fix verdict with no tools are finished answers.
    // An Implement (or open-plan) turn that still owes an edit is a cop-out
    // even on round zero; completing it lets the model repeat the same
    // "Want me to…?" recap every turn.
    if !facts.used_tools && !facts.wants_edit() {
        return Decision::Complete;
    }
    // Edited, tested green, stopped: whatever the model wrote, the step is
    // finished. An edit demand here would force filler edits, then plan_stall.
    if facts.ready_to_close_step() {
        return if can_continue(facts, Demand::ClosePlanStep) {
            continue_demand(facts, Demand::ClosePlanStep)
        } else {
            Decision::ClosePlanStep
        };
    }
    if facts.closed_step_without_edits() && !facts.wants_verify() {
        return Decision::Complete;
    }
    if !facts.visible_answer {
        if facts.wants_verify() && can_continue(facts, Demand::Verify) {
            return continue_demand(facts, Demand::Verify);
        }
        if facts.wants_edit() && can_continue(facts, Demand::Edit) {
            return continue_demand(facts, Demand::Edit);
        }
        if can_continue(facts, Demand::VisibleAnswer) {
            return continue_demand(facts, Demand::VisibleAnswer);
        }
        return Decision::Error {
            kind: "empty_stop",
            message: EMPTY_STOP_MSG,
        };
    }
    if facts.wants_verify() {
        if can_continue(facts, Demand::Verify) {
            return continue_demand(facts, Demand::Verify);
        }
        return unmet_error(facts);
    }
    if facts.wants_edit() {
        if can_continue(facts, Demand::Edit) {
            return continue_demand(facts, Demand::Edit);
        }
        if facts.plan_open && !facts.last_chance {
            facts.last_chance = true;
            return continue_demand(facts, Demand::Edit);
        }
        return unmet_error(facts);
    }
    Decision::Complete
}

fn decide_tools_finished(facts: &mut TurnFacts, inspect_repeat_capped: bool) -> Decision {
    if inspect_repeat_capped {
        return capped_repeat_demand(facts);
    }
    let demand = stall_demand(facts);
    // A bare Verdict is owed only on Review, or on Fix once tests ran with
    // no plan open. Unique-file reads and same-file paging are then the
    // task itself, so they never spend the stall budget. Exact repeats
    // still arrive as `inspect_repeat_capped` above, and an empty stop
    // after tools is still caught by `decide_model_stop`.
    if demand == Demand::Verdict {
        return Decision::Proceed;
    }
    let pressure = facts.stall_pressure();
    if pressure < STALL_ROUNDS_BEFORE_DEMAND {
        return Decision::Proceed;
    }
    if pressure < STALL_ROUNDS_BEFORE_ERROR {
        return continue_demand(facts, demand);
    }
    // The inspect that hit the cap often *is* the last file the model
    // needed (live ~/chat: plan + paginated server/db/state, then ws.rs).
    // Give one edit demand with that result in context before plan_stall.
    if facts.wants_edit() && facts.plan_open && !facts.last_chance {
        facts.last_chance = true;
        return continue_demand(facts, demand);
    }
    out_of_demands(facts)
}

fn capped_repeat_demand(facts: &mut TurnFacts) -> Decision {
    let demand = stall_demand(facts);
    if can_continue(facts, demand) {
        return continue_demand(facts, demand);
    }
    if !facts.last_chance {
        facts.last_chance = true;
        return continue_demand(facts, demand);
    }
    out_of_demands(facts)
}

/// The stall budget is spent. A finished step closes instead of erroring.
fn out_of_demands(facts: &TurnFacts) -> Decision {
    if facts.ready_to_close_step() {
        Decision::ClosePlanStep
    } else {
        unmet_error(facts)
    }
}

fn unmet_error(facts: &TurnFacts) -> Decision {
    if facts.wants_edit() && facts.plan_open {
        if facts.closed_step_without_edits() {
            return Decision::Complete;
        }
        return Decision::Error {
            kind: "plan_stall",
            message: if facts.step_kept_open {
                PLAN_KEPT_OPEN_MSG
            } else if facts.step_work && facts.verify_failed {
                PLAN_STALL_FAILING_MSG
            } else {
                PLAN_STALL_MSG
            },
        };
    }
    // The tree did change; only the test run is missing. Saying "without
    // applying the requested changes" here would contradict the diff the
    // user can see.
    if facts.mutated && facts.wants_verify() {
        return Decision::Error {
            kind: "unverified_stop",
            message: UNVERIFIED_STOP_MSG,
        };
    }
    if facts.wants_edit() || facts.wants_verify() {
        return Decision::Error {
            kind: "inspect_stop",
            message: INSPECT_STOP_MSG,
        };
    }
    Decision::Error {
        kind: "empty_stop",
        message: EMPTY_STOP_MSG,
    }
}

pub fn stall_demand(facts: &TurnFacts) -> Demand {
    if facts.wants_verify() {
        Demand::Verify
    } else if facts.ready_to_close_step() {
        Demand::ClosePlanStep
    } else if facts.wants_edit() {
        Demand::Edit
    } else {
        Demand::Verdict
    }
}

fn can_continue(facts: &TurnFacts, demand: Demand) -> bool {
    match demand {
        Demand::VisibleAnswer => facts.empty_after_tools < EMPTY_AFTER_TOOLS_CONTINUATIONS,
        other => {
            if facts.last_demand == Some(other) {
                facts.demand_continuations < DEMAND_CONTINUATIONS
            } else {
                true
            }
        }
    }
}

fn continue_demand(facts: &TurnFacts, demand: Demand) -> Decision {
    Decision::Continue {
        demand,
        hint: hint_for_facts(facts, demand),
    }
}

pub fn hint_for_facts(facts: &TurnFacts, demand: Demand) -> &'static str {
    match demand {
        Demand::Edit if facts.failed_mutation => FAILED_EDIT_HINT,
        Demand::Edit if facts.step_work && facts.verify_failed => FAILING_TESTS_HINT,
        Demand::Edit if facts.test_close_available() => PLAN_EXECUTE_HINT,
        Demand::Edit if facts.closed_step_without_edits() => NEXT_STEP_HINT,
        Demand::Edit => EDIT_NOW_HINT,
        Demand::Verify => UNVERIFIED_FIX_HINT,
        Demand::Verdict => INSPECT_STOP_VERDICT_HINT,
        Demand::VisibleAnswer => EMPTY_AFTER_TOOLS_HINT,
        Demand::ClosePlanStep => PLAN_CLOSE_HINT,
    }
}

pub fn status_for_facts(facts: &TurnFacts, demand: Demand) -> &'static str {
    match demand {
        Demand::Edit if facts.failed_mutation => {
            "last edit missed; asking to read then edit or close the step"
        }
        Demand::Edit if facts.step_work && facts.verify_failed => {
            "tests failed after the edit; asking for a fix"
        }
        Demand::Edit => "task in progress; asking for an edit",
        Demand::Verify => "review-and-fix stopped after inspection; asking for tests",
        Demand::Verdict => "inspect-repeat stop; asking for a verdict",
        Demand::VisibleAnswer => "empty model stop after tools; continuing",
        Demand::ClosePlanStep => "step edited and tests passed; asking to mark it done",
    }
}

/// TypeSafe may only change Edit/Verify/Verdict flavor. `Inspect` is ignored.
///
/// An open plan that has not been edited yet cannot become tests or a
/// verdict: that is the live ~/chat "lets build all of that" stall (cargo
/// test of the current tree, inspect-repeat, then `plan_stall`). Remaining
/// plan steps are also not a cop-out after cargo test.
pub fn flavor_demand(current: Demand, typesafe: crate::NextAction, facts: &TurnFacts) -> Demand {
    if matches!(current, Demand::VisibleAnswer | Demand::ClosePlanStep)
        || typesafe == crate::NextAction::Inspect
    {
        return current;
    }
    if facts.plan_open && !facts.mutated && current == Demand::Edit {
        return Demand::Edit;
    }
    if facts.plan_open && typesafe == crate::NextAction::Verdict {
        return current;
    }
    match typesafe {
        crate::NextAction::Inspect => current,
        crate::NextAction::Edit => Demand::Edit,
        crate::NextAction::Verify => Demand::Verify,
        crate::NextAction::Verdict => Demand::Verdict,
    }
}

pub fn demand_from_typesafe(action: crate::NextAction) -> Option<Demand> {
    match action {
        crate::NextAction::Inspect => None,
        crate::NextAction::Edit => Some(Demand::Edit),
        crate::NextAction::Verify => Some(Demand::Verify),
        crate::NextAction::Verdict => Some(Demand::Verdict),
    }
}

pub fn latest_prompt_text(messages: &[Message], input: &str) -> String {
    let trimmed = input.trim();
    if !trimmed.is_empty() {
        return trimmed.to_string();
    }
    messages
        .iter()
        .rev()
        .find(|message| message.role == hi_ai::Role::User)
        .map(|message| message.text())
        .unwrap_or_default()
}

/// Short "keep going" lines. Punctuation is ignored so
/// [`crate::CONTINUE_PLAN_PROMPT`] matches. A drive prompt that starts
/// with "Continue the open plan." plus the active step still counts;
/// a bare "continue using …" does not.
pub fn is_continuation_prompt(text: &str) -> bool {
    let mut normalized = String::new();
    for ch in text.chars() {
        if ch.is_ascii_punctuation() {
            continue;
        }
        if ch.is_whitespace() {
            if !normalized.is_empty() && !normalized.ends_with(' ') {
                normalized.push(' ');
            }
            continue;
        }
        normalized.push(ch.to_ascii_lowercase());
    }
    let trimmed = normalized.trim();
    matches!(
        trimmed,
        "continue"
            | "continue the open plan"
            | "keep going"
            | "go ahead"
            | "proceed"
            | "do it"
            | "apply it"
            | "apply those"
            | "apply that"
            | "yes"
            | "y"
            | "ok"
            | "okay"
    ) || trimmed.starts_with("continue the open plan ")
}

fn is_weak_ack(text: &str) -> bool {
    let mut normalized = String::new();
    for ch in text.chars() {
        if ch.is_ascii_punctuation() || ch.is_whitespace() {
            continue;
        }
        normalized.push(ch.to_ascii_lowercase());
    }
    matches!(normalized.as_str(), "yes" | "y" | "ok" | "okay")
}

/// Intent for this turn. A prompt that names the work itself (`fix`,
/// `apply those`) keeps that intent. `"continue"` with an open plan is
/// Implement so the model cannot narrate the next patch and stop. Bare
/// continue/keep-going without a plan inherits the last Fix/Implement user
/// line. Weak acks (`yes`) only upgrade when a plan is already open.
pub fn intent_for_turn(messages: &[Message], input: &str, plan_open: bool) -> Intent {
    let prompt = latest_prompt_text(messages, input);
    let direct = Intent::from_prompt(&prompt);
    if direct != Intent::Review || !is_continuation_prompt(&prompt) {
        return direct;
    }
    if plan_open {
        return Intent::Implement;
    }
    if is_weak_ack(&prompt) {
        return Intent::Review;
    }
    let current = prompt.trim();
    for message in messages.iter().rev() {
        if message.role != hi_ai::Role::User {
            continue;
        }
        let text = message.text();
        if text.trim() == current
            || is_continuation_prompt(&text)
            || text.trim_start().starts_with("[hi:")
        {
            continue;
        }
        let prior = Intent::from_prompt(&text);
        if prior != Intent::Review {
            return prior;
        }
    }
    Intent::Review
}

pub fn user_asked_to_fix(text: &str) -> bool {
    text.to_ascii_lowercase()
        .split(|ch: char| !ch.is_ascii_alphabetic())
        .any(|word| matches!(word, "fix" | "fixes" | "repair" | "repairs"))
}

pub fn user_asked_to_implement(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    const NEGATIONS: &[&str] = &[
        "do not implement",
        "don't implement",
        "dont implement",
        "do not edit",
        "don't edit",
        "dont edit",
        "without implementing",
        "without editing",
        "do not make changes",
        "don't make changes",
        "dont make changes",
        "no code changes",
    ];
    if NEGATIONS.iter().any(|phrase| lower.contains(phrase)) {
        return false;
    }
    const PHRASES: &[&str] = &[
        "do all of that",
        "do all that",
        "build all of that",
        "start building",
        "building all of that",
        "building all that",
        "make those changes",
        "apply those",
        "implement that",
        "implement those",
        "implement it",
        "build out",
    ];
    if PHRASES.iter().any(|phrase| lower.contains(phrase)) {
        return true;
    }
    let words: Vec<&str> = lower
        .split(|ch: char| !ch.is_ascii_alphabetic())
        .filter(|word| !word.is_empty())
        .collect();
    words.iter().any(|word| {
        matches!(
            *word,
            "implement" | "implements" | "impliment" | "impliments"
        )
    }) || finish_the_implementation(&words)
}

/// Live ~/drme: "finish the av1 implimentation", sent five times, was a
/// Review turn, so a stale recap completed it with no edit owed. The verb
/// must lead: "is the implementation complete?" is still a question.
fn finish_the_implementation(words: &[&str]) -> bool {
    const LEADS: &[&str] = &[
        "please", "pls", "now", "ok", "okay", "so", "just", "go", "ahead", "and", "can", "could",
        "you", "lets", "let", "s",
    ];
    let verb = words.iter().find(|word| !LEADS.contains(word));
    matches!(verb, Some(&("finish" | "complete" | "build")))
        && words
            .iter()
            .any(|word| *word == "impl" || word.starts_with("implem") || word.starts_with("implim"))
}

pub fn plan_is_open(plan: &[PlanStep]) -> bool {
    !plan.is_empty() && !PlanStep::all_complete(plan)
}

/// Not for the lone step of a plan nothing has edited: "post a one-step
/// plan, run the tests" must not stand in for an Implement request.
fn tests_may_close(plan_open: bool, step_edited: bool, plan_steps: usize) -> bool {
    plan_open && (step_edited || plan_steps >= 2)
}

/// [`TurnFacts::tests_may_close_step`] for a stored plan, where
/// `step_edited` is [`crate::PlanDrive::step_edited`].
pub fn tests_may_close_plan_step(plan: &[PlanStep], step_edited: bool) -> bool {
    tests_may_close(plan_is_open(plan), step_edited, plan.len())
}

fn was_done(previous: &[PlanStep], step: &PlanStep) -> bool {
    previous
        .iter()
        .any(|prior| prior.title == step.title && prior.status == PlanStatus::Done)
}

/// `next` marks done a step that `previous` had not finished. Counting done
/// steps would miss a re-post that drops one done step and closes another.
pub fn closes_a_step(previous: &[PlanStep], next: &[PlanStep]) -> bool {
    next.iter()
        .any(|step| step.status == PlanStatus::Done && !was_done(previous, step))
}

/// `proposed` still lists the step `previous` had active as unfinished.
pub fn keeps_active_step_open(previous: &[PlanStep], proposed: &[PlanStep]) -> bool {
    let Some(active) = previous
        .iter()
        .find(|step| step.status == PlanStatus::Active)
        .or_else(|| previous.iter().find(|step| step.status != PlanStatus::Done))
    else {
        return false;
    };
    proposed
        .iter()
        .any(|step| step.title == active.title && step.status != PlanStatus::Done)
}

/// Keep `update_plan` honest: each newly `done` step spends one work credit.
/// Extra dones stay at their previous status (or pending). One edit plus
/// tests cannot close a long checklist.
pub fn accept_plan(
    previous: &[PlanStep],
    mut proposed: Vec<PlanStep>,
    work_credits: &mut u32,
) -> Vec<PlanStep> {
    for step in &mut proposed {
        if step.status != PlanStatus::Done || was_done(previous, step) {
            continue;
        }
        if *work_credits > 0 {
            *work_credits = work_credits.saturating_sub(1);
            continue;
        }
        step.status = previous
            .iter()
            .find(|prior| prior.title == step.title)
            .map(|prior| prior.status)
            .filter(|status| *status != PlanStatus::Done)
            .unwrap_or(PlanStatus::Pending);
    }
    proposed
}

/// Tool result for an `update_plan` that [`accept_plan`] partly refused.
/// The tool itself echoes the proposed statuses ("Plan recorded: 1/7
/// done."), so without this the model believes the step is closed while
/// every later demand still asks for it.
pub fn plan_refusal_note(
    proposed: &[PlanStep],
    accepted: &[PlanStep],
    facts: &TurnFacts,
) -> Option<String> {
    let kept_open: Vec<String> = proposed
        .iter()
        .zip(accepted)
        .filter(|(sent, kept)| sent.status == PlanStatus::Done && kept.status != PlanStatus::Done)
        .map(|(sent, _)| format!("\"{}\"", sent.title))
        .collect();
    if kept_open.is_empty() {
        return None;
    }
    let done = accepted
        .iter()
        .filter(|step| step.status == PlanStatus::Done)
        .count();
    let route = if facts.test_close_available() {
        "If one of them is already implemented in the tree, run the project's \
         tests; once they pass, call update_plan again to close that one step."
    } else {
        "Implement what is missing with edit/write, run the tests, then call \
         update_plan again."
    };
    Some(format!(
        "Plan recorded: {done}/{} done. Not marked done: {}. Each newly done step \
         needs an edit applied this turn; a passing test run can close one more \
         step per turn, unless it is the only step of a plan nothing has edited. \
         {route} Do not add filler edits.",
        accepted.len(),
        kept_open.join(", ")
    ))
}

/// `plan` with its active step (else the first unfinished one) marked done
/// and the next unfinished step made active, plus the closed step's title.
pub fn close_active_step(plan: &[PlanStep]) -> Option<(Vec<PlanStep>, String)> {
    let index = plan
        .iter()
        .position(|step| step.status == PlanStatus::Active)
        .or_else(|| plan.iter().position(|step| step.status != PlanStatus::Done))?;
    let mut next = plan.to_vec();
    next[index].status = PlanStatus::Done;
    if !next.iter().any(|step| step.status == PlanStatus::Active)
        && let Some(following) = (index + 1..next.len())
            .chain(0..index)
            .find(|&i| next[i].status != PlanStatus::Done)
    {
        next[following].status = PlanStatus::Active;
    }
    let title = next[index].title.clone();
    Some((next, title))
}

/// Last `update_plan` tool call in the transcript, if any.
pub fn plan_from_messages(messages: &[Message]) -> Vec<PlanStep> {
    let mut last = None;
    for message in messages {
        for block in &message.content {
            if let Content::ToolCall {
                name, arguments, ..
            } = block
                && name == "update_plan"
                && let Some(steps) = plan_steps_from_arguments(arguments)
            {
                last = Some(steps);
            }
        }
    }
    last.unwrap_or_default()
}

/// Runner + subcommand pairs that verify the tree. Flags or lifecycle
/// phases may sit between them (`mvn -q test`, `./gradlew clean test`,
/// `npm run test:unit`). A Fix turn is not done until one of these ran; a
/// project whose real runner is missing here is nagged for `cargo test`
/// and then errors, so the table is deliberately wide.
const VERIFY_RUNNERS: &[(&str, &[&str])] = &[
    ("cargo", &["test", "nextest", "check", "clippy", "build"]),
    ("npm", &["test"]),
    ("yarn", &["test"]),
    ("pnpm", &["test"]),
    ("bun", &["test"]),
    ("deno", &["test"]),
    ("npx", &["vitest", "jest", "mocha"]),
    ("go", &["test", "vet"]),
    ("make", &["test", "check"]),
    ("just", &["test"]),
    ("mvn", &["test", "verify"]),
    ("mvnw", &["test", "verify"]),
    ("gradle", &["test", "check"]),
    ("gradlew", &["test", "check"]),
    ("sbt", &["test"]),
    ("dotnet", &["test"]),
    ("rake", &["test"]),
    ("mix", &["test"]),
    ("zig", &["test"]),
    ("swift", &["test"]),
    ("dart", &["test"]),
    ("flutter", &["test"]),
    ("bazel", &["test"]),
    ("stack", &["test"]),
    ("cabal", &["test"]),
];

/// Runners that are a verify on their own (`pytest -q`, `python -m unittest`).
const VERIFY_TOKENS: &[&str] = &[
    "pytest", "unittest", "rspec", "ctest", "phpunit", "vitest", "jest",
];

pub fn looks_like_verify(name: &str, arguments: &str) -> bool {
    if !name.eq_ignore_ascii_case("bash") {
        return false;
    }
    let lower = bash_command(arguments).to_ascii_lowercase();
    shell_segments(&lower).any(segment_is_verify)
}

/// Failure lines that survive a pager. `cargo test 2>&1 | tail -5` exits 0
/// whatever the tests did, so the status alone would call it green.
const VERIFY_FAILURE_MARKERS: &[&str] = &[
    "test result: FAILED",
    "error: could not compile",
    "error: test failed",
    "Failures (read these first)",
    "--- FAIL:",
    "FAILED (failures=",
    "FAILED (errors=",
];

/// A verify run passed: the command succeeded and its output shows none
/// of the usual runner failure lines.
pub fn verify_passed(status: hi_tools::ToolStatus, content: &str) -> bool {
    status == hi_tools::ToolStatus::Succeeded
        && !VERIFY_FAILURE_MARKERS
            .iter()
            .any(|marker| content.contains(marker))
}

fn segment_is_verify(segment: &str) -> bool {
    let tokens: Vec<&str> = segment
        .split_whitespace()
        .filter(|token| *token != "sudo")
        .map(|token| token.rsplit('/').next().unwrap_or(token))
        .collect();
    if tokens.iter().any(|token| VERIFY_TOKENS.contains(token)) {
        return true;
    }
    VERIFY_RUNNERS.iter().any(|(runner, subcommands)| {
        let Some(at) = tokens.iter().position(|token| token == runner) else {
            return false;
        };
        tokens[at + 1..].iter().any(|token| {
            subcommands.iter().any(|sub| {
                *token == *sub
                    || token
                        .strip_prefix(*sub)
                        .is_some_and(|rest| rest.starts_with(':'))
            })
        })
    })
}

/// Pipeline segments of a shell line, so `cd build && ctest` finds the
/// runner in its own segment and `go run x && echo test` does not pair
/// `go` with a later `test`.
fn shell_segments(command: &str) -> impl Iterator<Item = &str> {
    command
        .split(['\n', ';', '|'])
        .flat_map(|part| part.split("&&"))
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
}

pub fn is_stall_tool(name: &str, arguments: &str) -> bool {
    if looks_like_verify(name, arguments) {
        return false;
    }
    is_inspect_tool(name) || is_coordination(name) || is_bash_source_inspect(name, arguments)
}

pub fn stall_tool_round(calls: &[crate::pipe::ToolCall]) -> bool {
    !calls.is_empty()
        && calls
            .iter()
            .all(|call| is_stall_tool(&call.name, &call.arguments))
}

pub fn coordination_only_round(calls: &[crate::pipe::ToolCall]) -> bool {
    !calls.is_empty() && calls.iter().all(|call| is_coordination(&call.name))
}

pub fn read_pagination_only(calls: &[crate::pipe::ToolCall], seen_paths: &[String]) -> bool {
    !calls.is_empty()
        && calls.iter().all(|call| {
            read_tool_path(&call.name, &call.arguments)
                .is_some_and(|path| seen_paths.iter().any(|seen| seen == &path))
        })
}

fn read_tool_path(name: &str, arguments: &str) -> Option<String> {
    if !name.eq_ignore_ascii_case("read") {
        return None;
    }
    hi_tools::catalog::target_path("read", arguments)
}

pub fn is_bash_source_inspect(name: &str, arguments: &str) -> bool {
    if !name.eq_ignore_ascii_case("bash") {
        return false;
    }
    if looks_like_verify(name, arguments) {
        return false;
    }
    let lower = bash_command(arguments).to_ascii_lowercase();
    if lower.contains("cargo ")
        || lower.contains("npm ")
        || lower.contains("npx ")
        || lower.contains("pytest")
        || lower.contains("go test")
    {
        return false;
    }
    const CMDS: &[&str] = &[
        "cat", "head", "tail", "sed", "less", "more", "rg", "grep", "awk", "nl", "od", "xxd",
    ];
    tokenize_shell(&lower).into_iter().any(|token| {
        let base = token.rsplit('/').next().unwrap_or(token);
        CMDS.contains(&base)
    })
}

fn bash_command(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("command")
                .and_then(|command| command.as_str())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| arguments.to_string())
}

fn tokenize_shell(command: &str) -> Vec<&str> {
    command
        .split(|ch: char| {
            ch.is_whitespace() || matches!(ch, '|' | ';' | '&' | '`' | '(' | ')' | '<' | '>' | '\n')
        })
        .filter(|token| !token.is_empty() && *token != "sudo")
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::Message;
    use hi_tools::PlanStatus;

    fn facts(intent: Intent) -> TurnFacts {
        TurnFacts::new(intent, false)
    }

    #[test]
    fn python_unittest_satisfies_the_test_execution_gate() {
        assert!(looks_like_verify(
            "bash",
            r#"{"command":"python3 -m unittest -v"}"#
        ));
        assert!(!looks_like_verify("read", r#"{"path":"unittest.py"}"#));
    }

    #[test]
    fn project_test_runners_beyond_cargo_count_as_verify() {
        for command in [
            "make test",
            "just test",
            "mvn -q test",
            "./gradlew test",
            "dotnet test --no-build",
            "bun test",
            "deno test -A",
            "pnpm test",
            "bundle exec rspec spec/",
            "mix test",
            "cd build && ctest --output-on-failure",
            "cargo nextest run",
            "zig test src/main.zig",
            "vendor/bin/phpunit",
            "npm run test:unit",
            "cargo fmt && cargo clippy -- -D warnings && cargo test",
        ] {
            let arguments = serde_json::json!({ "command": command }).to_string();
            assert!(
                looks_like_verify("bash", &arguments),
                "{command} must count as verification"
            );
            assert!(
                !is_stall_tool("bash", &arguments),
                "{command} is never a stall round"
            );
        }
        for command in [
            "echo ok",
            "ls -la",
            "cat src/main.rs",
            "git status",
            "go run main.go && echo test",
            "npm run lint",
        ] {
            let arguments = serde_json::json!({ "command": command }).to_string();
            assert!(!looks_like_verify("bash", &arguments), "{command}");
        }
    }

    #[test]
    fn edit_without_tests_ends_as_unverified_not_inspect_stop() {
        let mut edited = facts(Intent::Fix);
        edited.used_tools = true;
        edited.mutated = true;
        edited.ran_verify = false;
        edited.visible_answer = true;
        let first = decide(Event::ModelStop, &mut edited);
        assert!(is_continue(&first, Demand::Verify), "{first:?}");
        edited.note_continue(Demand::Verify);
        let second = decide(Event::ModelStop, &mut edited);
        assert!(
            is_error(&second, "unverified_stop"),
            "an applied edit must not be reported as 'without applying', got {second:?}"
        );
        // No mutation at all is still the inspect stop.
        let mut untouched = facts(Intent::Fix);
        untouched.used_tools = true;
        untouched.visible_answer = true;
        assert!(is_continue(
            &decide(Event::ModelStop, &mut untouched),
            Demand::Verify
        ));
        untouched.note_continue(Demand::Verify);
        assert!(is_error(
            &decide(Event::ModelStop, &mut untouched),
            "inspect_stop"
        ));
    }

    fn open_plan_facts(intent: Intent) -> TurnFacts {
        TurnFacts::new(intent, true)
    }

    fn is_complete(decision: &Decision) -> bool {
        matches!(decision, Decision::Complete)
    }

    fn is_error(decision: &Decision, kind: &str) -> bool {
        matches!(decision, Decision::Error { kind: k, .. } if *k == kind)
    }

    fn is_continue(decision: &Decision, demand: Demand) -> bool {
        matches!(decision, Decision::Continue { demand: d, .. } if *d == demand)
    }

    fn is_proceed(decision: &Decision) -> bool {
        matches!(decision, Decision::Proceed)
    }

    fn tools_finished(facts: &mut TurnFacts) -> Decision {
        decide(
            Event::ToolsFinished {
                inspect_repeat_capped: false,
            },
            facts,
        )
    }

    #[test]
    fn round_zero_empty_completes() {
        let mut facts = facts(Intent::Review);
        assert!(is_complete(&decide(Event::ModelStop, &mut facts)));
    }

    #[test]
    fn open_plan_text_only_recap_demands_edit() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.visible_answer = true;
        let first = decide(Event::ModelStop, &mut facts);
        assert!(
            is_continue(&first, Demand::Edit),
            "a text-only recap with an open plan must demand an edit, got {first:?}"
        );
        facts.note_continue(Demand::Edit);
        let second = decide(Event::ModelStop, &mut facts);
        assert!(
            is_continue(&second, Demand::Edit),
            "one last edit push after a text-only recap, got {second:?}"
        );
        facts.note_continue(Demand::Edit);
        let third = decide(Event::ModelStop, &mut facts);
        assert!(is_error(&third, "plan_stall"), "{third:?}");
    }

    #[test]
    fn review_text_only_still_completes() {
        let mut facts = facts(Intent::Review);
        facts.visible_answer = true;
        assert!(is_complete(&decide(Event::ModelStop, &mut facts)));
    }

    #[test]
    fn failed_edit_hint_allows_update_plan_and_grants_a_reconcile_credit() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.failed_mutation = true;
        facts.visible_answer = true;
        let first = decide(Event::ModelStop, &mut facts);
        match first {
            Decision::Continue { demand, hint } => {
                assert_eq!(demand, Demand::Edit);
                assert!(hint.contains("update_plan"), "{hint}");
                assert!(!hint.contains("Do not call update_plan"), "{hint}");
            }
            other => panic!("expected edit demand, got {other:?}"),
        }
        facts.grant_reconcile_credit();
        assert_eq!(facts.work_credits, 1);
        let previous = vec![hi_tools::PlanStep {
            title: "Frame id numbers in frame header/decoder".into(),
            status: PlanStatus::Active,
        }];
        let proposed = vec![
            hi_tools::PlanStep {
                title: "Frame id numbers in frame header/decoder".into(),
                status: PlanStatus::Done,
            },
            hi_tools::PlanStep {
                title: "Timing info read/write".into(),
                status: PlanStatus::Active,
            },
        ];
        let plan = accept_plan(&previous, proposed, &mut facts.work_credits);
        assert_eq!(plan[0].status, PlanStatus::Done);
        assert_eq!(facts.work_credits, 0);
    }

    #[test]
    fn tools_then_empty_never_completes() {
        let mut facts = facts(Intent::Review);
        facts.used_tools = true;
        facts.visible_answer = false;
        let first = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&first, Demand::VisibleAnswer), "{first:?}");
        facts.note_continue(Demand::VisibleAnswer);
        let second = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&second, Demand::VisibleAnswer), "{second:?}");
        facts.note_continue(Demand::VisibleAnswer);
        let third = decide(Event::ModelStop, &mut facts);
        assert!(is_error(&third, "empty_stop"), "{third:?}");
        assert!(!is_complete(&third));
    }

    #[test]
    fn reasoning_only_after_tools_is_not_a_visible_answer() {
        let mut facts = facts(Intent::Fix);
        facts.used_tools = true;
        facts.visible_answer = false;
        let decision = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&decision, Demand::Verify), "{decision:?}");
    }

    #[test]
    fn plan_then_offset_reads_never_complete_silently() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        facts.note_round(false, true, false);
        facts.note_round(false, true, false);
        assert_eq!(facts.consecutive_stall, 2);
        let first = tools_finished(&mut facts);
        assert!(is_continue(&first, Demand::Edit), "{first:?}");
        facts.note_continue(Demand::Edit);
        while facts.consecutive_stall + 1 < STALL_ROUNDS_BEFORE_ERROR {
            facts.note_round(false, true, false);
            let again = tools_finished(&mut facts);
            assert!(
                is_continue(&again, Demand::Edit),
                "keep demanding an edit until the hard stall cap, got {again:?} stall={}",
                facts.consecutive_stall
            );
            facts.note_continue(Demand::Edit);
        }
        facts.note_round(false, true, false);
        assert_eq!(facts.consecutive_stall, STALL_ROUNDS_BEFORE_ERROR);
        let last_chance = tools_finished(&mut facts);
        assert!(
            is_continue(&last_chance, Demand::Edit),
            "the inspect that hits the cap still gets one edit demand, got {last_chance:?}"
        );
        facts.note_continue(Demand::Edit);
        facts.note_round(false, true, false);
        let last = tools_finished(&mut facts);
        assert!(is_error(&last, "plan_stall"), "{last:?}");
        assert!(!is_complete(&last));
    }

    #[test]
    fn cargo_test_does_not_reset_stall_while_plan_open_without_edit() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        facts.note_round(false, true, false);
        assert_eq!(facts.consecutive_stall, 1);
        facts.ran_verify = true;
        facts.note_round(false, false, true);
        assert_eq!(
            facts.consecutive_stall, 1,
            "verify while plan_open && !mutated must keep the stall count"
        );
        facts.note_round(false, true, false);
        assert_eq!(facts.consecutive_stall, 2);
        let decision = tools_finished(&mut facts);
        assert!(is_continue(&decision, Demand::Edit), "{decision:?}");
    }

    #[test]
    fn opening_plan_is_not_a_stall_round() {
        let mut facts = facts(Intent::Implement);
        facts.used_tools = true;
        facts.note_round_kind(false, true, false, true, false);
        facts.plan_open = true;
        assert_eq!(
            facts.consecutive_stall, 0,
            "posting the checklist is progress"
        );
        assert!(facts.opening_plan_grace);
        facts.note_round_kind(false, true, false, true, false);
        assert_eq!(
            facts.consecutive_stall, 1,
            "later update_plan-only rounds still stall"
        );
    }

    #[test]
    fn live_chat_plan_plus_seven_reads_demands_edit() {
        let mut facts = facts(Intent::Implement);
        facts.used_tools = true;
        facts.note_round_kind(false, true, false, true, false);
        facts.plan_open = true;
        for _ in 0..7 {
            facts.note_round(false, true, false);
        }
        assert_eq!(facts.consecutive_stall, 7);
        let decision = tools_finished(&mut facts);
        assert!(
            is_continue(&decision, Demand::Edit),
            "plan + seven unique-file pages must keep demanding an edit, got {decision:?}"
        );
        assert!(!is_error(&decision, "plan_stall"));
    }

    fn read_call(id: &str, path: &str, offset: Option<u32>) -> crate::pipe::ToolCall {
        let arguments = match offset {
            Some(offset) => format!(r#"{{"path":"{path}","offset":{offset}}}"#),
            None => format!(r#"{{"path":"{path}"}}"#),
        };
        crate::pipe::ToolCall {
            id: id.into(),
            name: "read".into(),
            arguments,
        }
    }

    fn note_reads(facts: &mut TurnFacts, calls: &[crate::pipe::ToolCall]) {
        let pagination_only = read_pagination_only(calls, &facts.seen_read_paths);
        facts.observe_read_paths(calls);
        facts.note_round_kind(
            false,
            stall_tool_round(calls),
            false,
            coordination_only_round(calls),
            pagination_only,
        );
    }

    #[test]
    fn paginated_reads_of_the_same_file_do_not_spend_unique_stall_slots() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        note_reads(&mut facts, &[read_call("a", "src/protocol.rs", None)]);
        note_reads(&mut facts, &[read_call("b", "src/db.rs", None)]);
        assert_eq!(facts.consecutive_stall, 2);
        assert_eq!(facts.pagination_rounds, 0);
        let demand = tools_finished(&mut facts);
        assert!(is_continue(&demand, Demand::Edit), "{demand:?}");
        for (i, offset) in [200u32, 400, 700, 900, 1100].into_iter().enumerate() {
            note_reads(
                &mut facts,
                &[read_call(&format!("s{i}"), "src/server.rs", Some(offset))],
            );
        }
        // First server.rs page is a new path (stall 3); the rest are pagination.
        assert_eq!(facts.consecutive_stall, 3);
        assert_eq!(facts.pagination_rounds, 4);
        assert_eq!(facts.stall_pressure(), 4);
        let decision = tools_finished(&mut facts);
        assert!(
            is_continue(&decision, Demand::Edit),
            "paging one large file must not plan_stall, got {decision:?}"
        );
        assert!(!is_error(&decision, "plan_stall"));
    }

    #[test]
    fn live_chat_protocol_db_server_pages_do_not_plan_stall() {
        let mut facts = facts(Intent::Implement);
        facts.used_tools = true;
        facts.note_round_kind(false, true, false, true, false);
        facts.plan_open = true;
        note_reads(
            &mut facts,
            &[
                read_call("p", "src/protocol.rs", None),
                read_call("d", "src/db.rs", None),
            ],
        );
        note_reads(&mut facts, &[read_call("p2", "src/protocol.rs", Some(200))]);
        note_reads(&mut facts, &[read_call("d2", "src/db.rs", Some(200))]);
        note_reads(&mut facts, &[read_call("d3", "src/db.rs", Some(400))]);
        note_reads(&mut facts, &[read_call("s1", "src/server.rs", Some(200))]);
        for (i, offset) in [400u32, 700, 900, 1100].into_iter().enumerate() {
            note_reads(
                &mut facts,
                &[read_call(
                    &format!("s{}", i + 2),
                    "src/server.rs",
                    Some(offset),
                )],
            );
        }
        assert_eq!(facts.consecutive_stall, 2, "two unique-path rounds");
        // Opening server.rs restarts the page cap; only the four later
        // server.rs offsets count as pagination.
        assert_eq!(facts.pagination_rounds, 4);
        assert_eq!(facts.stall_pressure(), 4);
        let decision = tools_finished(&mut facts);
        assert!(
            is_continue(&decision, Demand::Edit),
            "live ~/chat paging protocol/db/server must keep demanding an edit, got {decision:?}"
        );
        assert!(!is_error(&decision, "plan_stall"));
    }

    #[test]
    fn same_file_pagination_still_hits_the_inspect_cap() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        note_reads(&mut facts, &[read_call("a", "src/server.rs", None)]);
        for i in 0..STALL_ROUNDS_BEFORE_ERROR {
            note_reads(
                &mut facts,
                &[read_call(
                    &format!("p{i}"),
                    "src/server.rs",
                    Some((i + 1) * 200),
                )],
            );
        }
        assert_eq!(facts.consecutive_stall, 1);
        assert_eq!(facts.pagination_rounds, STALL_ROUNDS_BEFORE_ERROR);
        let last_chance = tools_finished(&mut facts);
        assert!(
            is_continue(&last_chance, Demand::Edit),
            "the page that hits the cap still gets one edit demand, got {last_chance:?}"
        );
        note_reads(
            &mut facts,
            &[read_call("too_far", "src/server.rs", Some(9_000))],
        );
        let last = tools_finished(&mut facts);
        assert!(is_error(&last, "plan_stall"), "{last:?}");
    }

    #[test]
    fn inspect_repeat_empty_assistant_never_completes() {
        let mut facts = facts(Intent::Review);
        facts.used_tools = true;
        facts.visible_answer = false;
        facts.ran_verify = true;
        let first = decide(
            Event::ToolsFinished {
                inspect_repeat_capped: true,
            },
            &mut facts,
        );
        assert!(is_continue(&first, Demand::Verdict), "{first:?}");
        facts.note_continue(Demand::Verdict);
        let second = decide(
            Event::ToolsFinished {
                inspect_repeat_capped: true,
            },
            &mut facts,
        );
        assert!(
            is_continue(&second, Demand::Verdict),
            "re-hint once after inspect-repeat cap, got {second:?}"
        );
        facts.note_continue(Demand::Verdict);
        let third = decide(
            Event::ToolsFinished {
                inspect_repeat_capped: true,
            },
            &mut facts,
        );
        assert!(is_error(&third, "empty_stop"), "{third:?}");
        assert!(!is_complete(&third));
    }

    #[test]
    fn implement_prose_cop_out_after_inspect_asks_for_edit() {
        let mut facts = facts(Intent::Implement);
        facts.used_tools = true;
        facts.visible_answer = true;
        facts.note_round(false, true, false);
        let decision = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&decision, Demand::Edit), "{decision:?}");
        assert!(!is_complete(&decision));
        facts.note_continue(Demand::Edit);
        let again = decide(Event::ModelStop, &mut facts);
        assert!(is_error(&again, "inspect_stop"), "{again:?}");
        assert!(!is_complete(&again));
    }

    #[test]
    fn typesafe_inspect_cannot_skip_exhausted_budget() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        facts.note_round(false, true, false);
        facts.note_round(false, true, false);
        facts.note_continue(Demand::Edit);
        let first = tools_finished(&mut facts);
        assert!(is_continue(&first, Demand::Edit), "{first:?}");
        while facts.consecutive_stall + 1 < STALL_ROUNDS_BEFORE_ERROR {
            facts.note_round(false, true, false);
            let again = tools_finished(&mut facts);
            assert!(is_continue(&again, Demand::Edit), "{again:?}");
            facts.note_continue(Demand::Edit);
        }
        facts.note_round(false, true, false);
        let chance = tools_finished(&mut facts);
        assert!(
            is_continue(&chance, Demand::Edit),
            "exhausted inspect budget still gets one last edit demand, got {chance:?}"
        );
        facts.note_continue(Demand::Edit);
        facts.note_round(false, true, false);
        let decision = tools_finished(&mut facts);
        assert!(is_error(&decision, "plan_stall"), "{decision:?}");
        let flavored = flavor_demand(
            Demand::Edit,
            crate::NextAction::Inspect,
            &open_plan_facts(Intent::Implement),
        );
        assert_eq!(flavored, Demand::Edit);
    }

    #[test]
    fn typesafe_cannot_skip_unstarted_plan() {
        let facts = open_plan_facts(Intent::Implement);
        assert_eq!(
            flavor_demand(Demand::Edit, crate::NextAction::Verify, &facts),
            Demand::Edit,
            "cargo test of the current tree is not the next step on an unstarted plan"
        );
        assert_eq!(
            flavor_demand(Demand::Edit, crate::NextAction::Verdict, &facts),
            Demand::Edit,
            "a verdict cannot close unfinished plan steps"
        );
        assert_eq!(
            flavor_demand(Demand::Edit, crate::NextAction::Edit, &facts),
            Demand::Edit
        );
    }

    #[test]
    fn typesafe_cannot_verdict_remaining_plan_after_tests() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.mutated = true;
        facts.ran_verify = true;
        assert!(facts.wants_edit());
        assert_eq!(
            flavor_demand(Demand::Edit, crate::NextAction::Verdict, &facts),
            Demand::Edit
        );
    }

    #[test]
    fn typesafe_verify_still_flavors_review_and_fix() {
        let facts = facts(Intent::Fix);
        assert_eq!(
            flavor_demand(Demand::Verify, crate::NextAction::Verify, &facts),
            Demand::Verify
        );
        assert_eq!(
            flavor_demand(Demand::Verdict, crate::NextAction::Verify, &facts),
            Demand::Verify
        );
    }

    #[test]
    fn review_open_plan_does_not_demand_edit() {
        let mut facts = open_plan_facts(Intent::Review);
        facts.used_tools = true;
        facts.visible_answer = true;
        facts.note_round(false, true, false);
        facts.note_round(false, true, false);
        let decision = tools_finished(&mut facts);
        assert!(
            is_proceed(&decision),
            "a review keeps reading even with a plan open, got {decision:?}"
        );
        assert!(!is_continue(&decision, Demand::Edit));
        assert!(is_complete(&decide(Event::ModelStop, &mut facts)));
    }

    #[test]
    fn review_codebase_unique_reads_are_not_a_stall() {
        // Live ~/hi "review codebase": `list`, then `repo_map`, then the
        // harness said "Stop inspecting" and the model gave up with "tell
        // me what to do". Opening new files is the review.
        let mut facts = facts(Intent::Review);
        facts.used_tools = true;
        let paths = [
            "Cargo.toml",
            "crates/hi-harness/src/turn.rs",
            "crates/hi-harness/src/completion.rs",
            "crates/hi-tools/src/lib.rs",
            "crates/hi-tui/src/lib.rs",
            "crates/hi-cli/src/main.rs",
            "README.md",
            "docs/handbook.md",
            "crates/hi-harness/src/pipe.rs",
            "crates/hi-harness/src/tools.rs",
            "crates/hi-harness/src/ui.rs",
            "crates/hi-harness/src/prompt.rs",
        ];
        for (i, path) in paths.iter().enumerate() {
            note_reads(&mut facts, &[read_call(&format!("r{i}"), path, None)]);
            let decision = tools_finished(&mut facts);
            assert!(
                is_proceed(&decision),
                "unique read #{} of a review must proceed, got {decision:?}",
                i + 1
            );
        }
        assert!(
            facts.stall_pressure() > STALL_ROUNDS_BEFORE_ERROR,
            "the counter still runs; only the Verdict demand ignores it"
        );
        facts.visible_answer = true;
        assert!(is_complete(&decide(Event::ModelStop, &mut facts)));
    }

    #[test]
    fn review_paging_one_large_file_is_not_a_stall() {
        let mut facts = facts(Intent::Review);
        facts.used_tools = true;
        note_reads(&mut facts, &[read_call("a", "src/turn.rs", None)]);
        for i in 1..=(STALL_ROUNDS_BEFORE_ERROR + 4) {
            note_reads(
                &mut facts,
                &[read_call(&format!("p{i}"), "src/turn.rs", Some(i * 200))],
            );
            let decision = tools_finished(&mut facts);
            assert!(
                is_proceed(&decision),
                "page {i} of a review read must proceed, got {decision:?}"
            );
        }
    }

    #[test]
    fn review_identical_repeats_still_demand_a_verdict() {
        let mut facts = facts(Intent::Review);
        facts.used_tools = true;
        facts.note_round(false, true, false);
        let capped = decide(
            Event::ToolsFinished {
                inspect_repeat_capped: true,
            },
            &mut facts,
        );
        assert!(is_continue(&capped, Demand::Verdict), "{capped:?}");
    }

    #[test]
    fn fix_after_tests_without_plan_explores_freely() {
        // "review for any major issues and fix": the Verify demand made
        // the model run cargo test; with tests green and no plan there is
        // nothing concrete to demand, so it may keep looking for issues.
        let mut facts = facts(Intent::Fix);
        facts.used_tools = true;
        facts.ran_verify = true;
        for i in 0..STALL_ROUNDS_BEFORE_ERROR + 2 {
            facts.note_round(false, true, false);
            let decision = tools_finished(&mut facts);
            assert!(
                is_proceed(&decision),
                "post-test inspect #{} on a fix prompt must proceed, got {decision:?}",
                i + 1
            );
        }
    }

    #[test]
    fn fix_before_tests_still_demands_verify_after_two_inspects() {
        let mut facts = facts(Intent::Fix);
        facts.used_tools = true;
        facts.note_round(false, true, false);
        assert!(is_proceed(&tools_finished(&mut facts)));
        facts.note_round(false, true, false);
        let decision = tools_finished(&mut facts);
        assert!(is_continue(&decision, Demand::Verify), "{decision:?}");
    }

    #[test]
    fn review_inspect_then_verdict_completes() {
        let mut facts = facts(Intent::Review);
        facts.used_tools = true;
        facts.visible_answer = true;
        facts.ran_verify = true;
        assert!(is_complete(&decide(Event::ModelStop, &mut facts)));
    }

    #[test]
    fn edit_then_verify_then_verdict_completes() {
        let mut facts = facts(Intent::Implement);
        facts.used_tools = true;
        facts.mutated = true;
        facts.ran_verify = true;
        facts.visible_answer = true;
        assert!(is_complete(&decide(Event::ModelStop, &mut facts)));
    }

    #[test]
    fn implement_open_plan_cop_out_after_partial_edit_does_not_complete() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        facts.mutated = true;
        facts.ran_verify = true;
        facts.visible_answer = true;
        let first = decide(Event::ModelStop, &mut facts);
        assert!(
            is_continue(&first, Demand::Edit),
            "remaining plan steps must demand another edit, got {first:?}"
        );
        assert!(!is_complete(&first));
        facts.note_continue(Demand::Edit);
        let second = decide(Event::ModelStop, &mut facts);
        assert!(
            is_continue(&second, Demand::Edit),
            "one last edit push after a mid-plan cop-out, got {second:?}"
        );
        facts.note_continue(Demand::Edit);
        let third = decide(Event::ModelStop, &mut facts);
        assert!(is_error(&third, "plan_stall"), "{third:?}");
        assert!(!is_complete(&third));
    }

    #[test]
    fn post_edit_open_plan_asks_for_tests_before_more_edits() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        facts.mutated = true;
        facts.ran_verify = false;
        facts.visible_answer = true;
        let decision = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&decision, Demand::Verify), "{decision:?}");
    }

    #[test]
    fn post_edit_implement_asks_for_tests() {
        let mut facts = facts(Intent::Implement);
        facts.used_tools = true;
        facts.mutated = true;
        facts.ran_verify = false;
        facts.visible_answer = true;
        let decision = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&decision, Demand::Verify), "{decision:?}");
    }

    #[test]
    fn mixed_read_and_bash_cat_is_a_stall_round() {
        let calls = vec![
            crate::pipe::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: r#"{"path":"src/web.rs","offset":60}"#.into(),
            },
            crate::pipe::ToolCall {
                id: "b".into(),
                name: "bash".into(),
                arguments: r#"{"command":"cat src/web.rs"}"#.into(),
            },
        ];
        assert!(stall_tool_round(&calls));
        assert!(!is_stall_tool(
            "bash",
            r#"{"command":"cargo test --offline"}"#
        ));
        assert!(!is_stall_tool("bash", r#"{"command":"echo ok"}"#));
    }

    #[test]
    fn plan_from_messages_uses_last_update_plan() {
        let first = serde_json::json!({
            "steps": [{"title":"one","status":"active"}]
        })
        .to_string();
        let second = serde_json::json!({
            "steps": [
                {"title":"one","status":"done"},
                {"title":"two","status":"active"}
            ]
        })
        .to_string();
        let messages = vec![
            Message::assistant(vec![Content::ToolCall {
                id: "p1".into(),
                name: "update_plan".into(),
                arguments: first,
            }]),
            Message::assistant(vec![Content::ToolCall {
                id: "p2".into(),
                name: "update_plan".into(),
                arguments: second,
            }]),
        ];
        let plan = plan_from_messages(&messages);
        assert!(plan_is_open(&plan));
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].status, PlanStatus::Done);
        assert_eq!(plan[1].status, PlanStatus::Active);
    }

    #[test]
    fn intent_from_latest_prompt_only() {
        assert_eq!(
            Intent::from_prompt("review for any major issues and fix"),
            Intent::Fix
        );
        assert_eq!(Intent::from_prompt("do all of that"), Intent::Implement);
        // Live ~/drme: "start building all of that" with an open plan was
        // classified as review, so inspect rounds never hit plan_stall.
        assert_eq!(
            Intent::from_prompt("start building all of that"),
            Intent::Implement
        );
        assert_eq!(
            Intent::from_prompt(
                "How should we add GET /metrics? Post a step-by-step plan. Do not implement yet."
            ),
            Intent::Review
        );
        assert_eq!(
            Intent::from_prompt("how can we improve this"),
            Intent::Review
        );
        assert!(!user_asked_to_fix(
            "Run cargo test to verify the review change didn't break anything."
        ));
        // Live ~/drme prompts, typos included, ran as Review turns.
        for prompt in [
            "finish the av1 implimentation",
            "build out full av1 spec implimentation",
            "impliment the obu parser",
            "please complete the implementation",
            "can you finish the av1 impl",
        ] {
            assert_eq!(Intent::from_prompt(prompt), Intent::Implement, "{prompt}");
        }
        for prompt in [
            "is the implementation complete?",
            "review the av1 implementation",
            "how does the build work",
            "build the project and list the warnings",
        ] {
            assert_eq!(Intent::from_prompt(prompt), Intent::Review, "{prompt}");
        }
    }

    #[test]
    fn start_building_an_open_plan_demands_an_edit() {
        let mut facts = open_plan_facts(Intent::from_prompt("start building all of that"));
        facts.used_tools = true;
        facts.note_round(false, true, false);
        facts.note_round(false, true, false);
        let decision = tools_finished(&mut facts);
        assert!(
            is_continue(&decision, Demand::Edit),
            "inspecting instead of editing the open plan must demand an edit, got {decision:?}"
        );
    }

    #[test]
    fn continuation_prompt_with_open_plan_is_implement() {
        assert!(is_continuation_prompt("continue"));
        assert!(is_continuation_prompt("Continue."));
        assert!(is_continuation_prompt(crate::CONTINUE_PLAN_PROMPT));
        assert!(is_continuation_prompt(
            "Continue the open plan. Implement this step now with edit or write: Encoder-side superres."
        ));
        assert!(is_continuation_prompt("keep going"));
        assert!(!is_continuation_prompt("how can we improve this"));
        assert!(!is_continuation_prompt("continue using solana-sdk"));
        let older = vec![Message::user("review for any major issues and fix")];
        assert_eq!(intent_for_turn(&older, "continue", true), Intent::Implement);
        assert_eq!(intent_for_turn(&older, "continue", false), Intent::Fix);
        assert_eq!(
            intent_for_turn(&older, crate::CONTINUE_PLAN_PROMPT, true),
            Intent::Implement
        );
        let active = [hi_tools::PlanStep {
            title: "Encoder-side superres downscale + signaling".into(),
            status: hi_tools::PlanStatus::Active,
        }];
        for (stall, step_edited) in [(0, false), (0, true), (1, false), (1, true)] {
            let driven = crate::drive_continue_prompt(&active, stall, step_edited);
            assert_eq!(
                intent_for_turn(&older, &driven, true),
                Intent::Implement,
                "{driven}"
            );
            assert_eq!(driven.contains("update_plan"), step_edited, "{driven}");
            assert!(driven.contains("filler edits"), "{driven}");
        }
        let two_steps = [
            active[0].clone(),
            hi_tools::PlanStep {
                title: "Update README support matrix".into(),
                status: hi_tools::PlanStatus::Pending,
            },
        ];
        let driven = crate::drive_continue_prompt(&two_steps, 0, false);
        assert!(driven.contains("update_plan"), "{driven}");
        assert_eq!(
            intent_for_turn(&older, "how can we improve this", true),
            Intent::Review
        );
        assert_eq!(intent_for_turn(&older, "yes", true), Intent::Implement);
        assert_eq!(intent_for_turn(&older, "yes", false), Intent::Review);
        // An explicit implement phrase is never downgraded by the
        // continuation path, even after a review-only prompt with no plan.
        let review_only = vec![Message::user("how can we improve this")];
        assert_eq!(
            intent_for_turn(&review_only, "apply those", false),
            Intent::Implement
        );
        assert_eq!(
            intent_for_turn(&review_only, "continue", false),
            Intent::Review
        );
    }

    #[test]
    fn fix_open_plan_after_edit_and_verify_still_asks_for_edit() {
        let mut facts = open_plan_facts(Intent::Fix);
        facts.used_tools = true;
        facts.mutated = true;
        facts.ran_verify = true;
        facts.visible_answer = true;
        let decision = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&decision, Demand::Edit), "{decision:?}");
        assert!(!is_complete(&decision));
    }

    #[test]
    fn continue_open_plan_model_stop_demands_edit() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        facts.visible_answer = true;
        let decision = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&decision, Demand::Edit), "{decision:?}");
        assert!(!is_complete(&decision));
    }

    #[test]
    fn mutation_resets_stall() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.note_round(false, true, false);
        facts.note_round(false, true, false);
        assert_eq!(facts.consecutive_stall, 2);
        facts.mutated = true;
        facts.note_round(true, false, false);
        assert_eq!(facts.consecutive_stall, 0);
    }

    fn step(title: &str, status: PlanStatus) -> PlanStep {
        PlanStep {
            title: title.into(),
            status,
        }
    }

    #[test]
    fn accept_plan_one_edit_cannot_close_eight_steps() {
        let previous = vec![
            step("pool", PlanStatus::Active),
            step("history", PlanStatus::Pending),
            step("metrics", PlanStatus::Pending),
        ];
        let proposed = vec![
            step("pool", PlanStatus::Done),
            step("history", PlanStatus::Done),
            step("metrics", PlanStatus::Done),
        ];
        let mut credits = 2; // one edit + one verify
        let accepted = accept_plan(&previous, proposed, &mut credits);
        assert_eq!(accepted[0].status, PlanStatus::Done);
        assert_eq!(accepted[1].status, PlanStatus::Done);
        assert_eq!(accepted[2].status, PlanStatus::Pending);
        assert!(plan_is_open(&accepted));
        assert_eq!(credits, 0);
    }

    #[test]
    fn accept_plan_zero_credits_rejects_new_dones() {
        let previous = vec![step("pool", PlanStatus::Active)];
        let proposed = vec![step("pool", PlanStatus::Done)];
        let mut credits = 0;
        let accepted = accept_plan(&previous, proposed, &mut credits);
        assert_eq!(accepted[0].status, PlanStatus::Active);
        assert!(plan_is_open(&accepted));
    }

    #[test]
    fn dishonest_all_done_after_partial_edit_still_wants_edit() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        facts.mutated = true;
        facts.ran_verify = true;
        facts.visible_answer = true;
        facts.grant_mutation_credit();
        facts.grant_verify_credit();
        let previous = vec![
            step("pool", PlanStatus::Active),
            step("history", PlanStatus::Pending),
            step("rate limit", PlanStatus::Pending),
            step("headers", PlanStatus::Pending),
        ];
        let proposed = previous
            .iter()
            .map(|row| step(&row.title, PlanStatus::Done))
            .collect();
        let plan = accept_plan(&previous, proposed, &mut facts.work_credits);
        facts.plan_open = plan_is_open(&plan);
        assert!(facts.plan_open, "bulk close must leave unfinished steps");
        let first = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&first, Demand::Edit), "{first:?}");
        assert!(!is_complete(&first));
    }

    /// The active step, edited and tested green this turn.
    fn finished_step_facts() -> TurnFacts {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        facts.visible_answer = true;
        facts.mutated = true;
        facts.step_work = true;
        facts.grant_mutation_credit();
        facts.ran_verify = true;
        facts.grant_verify_credit();
        facts
    }

    /// Live ~/drme: edit + passing `cargo test` + "the step is complete"
    /// got "Your next action MUST be `edit`… Do not call update_plan" until
    /// plan_stall, and the model padded the file with comment lines.
    #[test]
    fn finished_step_asks_for_update_plan_then_closes_itself() {
        let mut facts = finished_step_facts();
        assert!(facts.ready_to_close_step());
        match decide(Event::ModelStop, &mut facts) {
            Decision::Continue { demand, hint } => {
                assert_eq!(demand, Demand::ClosePlanStep);
                assert!(hint.contains("call update_plan now"), "{hint}");
                assert!(hint.contains("filler"), "{hint}");
            }
            other => panic!("a finished step must ask for update_plan, got {other:?}"),
        }
        facts.note_continue(Demand::ClosePlanStep);
        assert_eq!(
            decide(Event::ModelStop, &mut facts),
            Decision::ClosePlanStep,
            "an ignored close hint closes the step instead of plan_stall"
        );
        let mut empty = finished_step_facts();
        empty.visible_answer = false;
        assert!(is_continue(
            &decide(Event::ModelStop, &mut empty),
            Demand::ClosePlanStep
        ));
    }

    #[test]
    fn finished_step_closes_at_the_stall_cap_instead_of_plan_stall() {
        let mut facts = finished_step_facts();
        facts.consecutive_stall = STALL_ROUNDS_BEFORE_DEMAND;
        assert!(is_continue(
            &tools_finished(&mut facts),
            Demand::ClosePlanStep
        ));
        facts.consecutive_stall = STALL_ROUNDS_BEFORE_ERROR;
        let last = tools_finished(&mut facts);
        assert!(is_continue(&last, Demand::ClosePlanStep), "{last:?}");
        assert_eq!(tools_finished(&mut facts), Decision::ClosePlanStep);

        let mut capped = finished_step_facts();
        let capped_decide = |facts: &mut TurnFacts| {
            decide(
                Event::ToolsFinished {
                    inspect_repeat_capped: true,
                },
                facts,
            )
        };
        assert!(is_continue(
            &capped_decide(&mut capped),
            Demand::ClosePlanStep
        ));
        capped.note_continue(Demand::ClosePlanStep);
        assert!(is_continue(
            &capped_decide(&mut capped),
            Demand::ClosePlanStep
        ));
        assert_eq!(capped_decide(&mut capped), Decision::ClosePlanStep);
    }

    /// Live ~/drme turn 53: after green tests the model answered the close
    /// demand with `update_plan` keeping "Verify dav1d + ffmpeg + tests"
    /// active because dav1d still rejected the stream. The harness closed
    /// it anyway, the plan read 4/4, and nothing resumed the work.
    #[test]
    fn a_step_the_model_keeps_open_is_not_closed_for_it() {
        let previous = vec![
            step("Parse sequence header", PlanStatus::Done),
            step("Write frame OBUs", PlanStatus::Done),
            step("Encode tiles", PlanStatus::Done),
            step("Verify dav1d + ffmpeg + tests", PlanStatus::Active),
        ];
        let plan_stall = Decision::Error {
            kind: "plan_stall",
            message: PLAN_KEPT_OPEN_MSG,
        };
        let mut facts = finished_step_facts();
        facts.note_plan_update(&previous, &previous);
        assert!(!facts.ready_to_close_step());
        let first = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&first, Demand::Edit), "{first:?}");
        facts.note_continue(Demand::Edit);
        let last = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&last, Demand::Edit), "{last:?}");
        facts.note_continue(Demand::Edit);
        assert_eq!(decide(Event::ModelStop, &mut facts), plan_stall);

        let mut stalled = finished_step_facts();
        stalled.note_plan_update(&previous, &previous);
        stalled.consecutive_stall = STALL_ROUNDS_BEFORE_ERROR;
        stalled.last_chance = true;
        assert_eq!(tools_finished(&mut stalled), plan_stall);

        let mut done = previous.clone();
        done[3].status = PlanStatus::Done;
        facts.note_plan_update(&previous, &done);
        assert!(
            !facts.step_kept_open,
            "marking it done is not keeping it open"
        );

        // Posted before the tests ran, the plan says nothing about them.
        let mut early = finished_step_facts();
        early.ran_verify = false;
        early.note_plan_update(&previous, &previous);
        early.ran_verify = true;
        assert!(early.ready_to_close_step());
    }

    #[test]
    fn failing_tests_keep_the_step_open_and_plan_stall_says_why() {
        let mut facts = finished_step_facts();
        facts.verify_failed = true;
        assert!(!facts.ready_to_close_step());
        let first = decide(Event::ModelStop, &mut facts);
        assert_eq!(
            first,
            Decision::Continue {
                demand: Demand::Edit,
                hint: FAILING_TESTS_HINT
            }
        );
        facts.note_continue(Demand::Edit);
        let second = decide(Event::ModelStop, &mut facts);
        assert!(is_continue(&second, Demand::Edit), "{second:?}");
        facts.note_continue(Demand::Edit);
        match decide(Event::ModelStop, &mut facts) {
            Decision::Error { kind, message } => {
                assert_eq!(kind, "plan_stall");
                assert!(message.contains("tests failed"), "{message}");
            }
            other => panic!("expected plan_stall, got {other:?}"),
        }
    }

    #[test]
    fn step_without_an_edit_is_never_closed_by_the_harness() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        facts.visible_answer = true;
        facts.plan_steps = 2;
        facts.ran_verify = true;
        facts.grant_verify_credit();
        assert_eq!(facts.work_credits, 1, "a test run may close one step");
        assert!(
            !facts.ready_to_close_step(),
            "but only the model may say so"
        );
        let first = decide(Event::ModelStop, &mut facts);
        match first {
            Decision::Continue { demand, hint } => {
                assert_eq!(demand, Demand::Edit);
                assert_eq!(hint, PLAN_EXECUTE_HINT);
            }
            other => panic!("expected an edit demand, got {other:?}"),
        }
        let mut review = finished_step_facts();
        review.intent = Intent::Review;
        assert!(!review.ready_to_close_step());
    }

    #[test]
    fn passing_tests_close_one_step_per_turn_but_never_a_lone_untouched_step() {
        let mut lone = open_plan_facts(Intent::Implement);
        lone.plan_steps = 1;
        lone.grant_verify_credit();
        assert_eq!(lone.work_credits, 0, "a one-step plan needs an edit");

        let mut baseline = open_plan_facts(Intent::Implement);
        baseline.plan_steps = 3;
        baseline.grant_verify_credit();
        assert_eq!(
            baseline.work_credits, 1,
            "\"Baseline: run cargo test\" closes on the run"
        );

        let mut carried = open_plan_facts(Intent::Implement);
        carried.plan_steps = 1;
        carried.step_work = true;
        carried.grant_verify_credit();
        assert_eq!(carried.work_credits, 1);
        carried.grant_verify_credit();
        assert_eq!(carried.work_credits, 1, "one test-run credit per turn");

        let mut red = open_plan_facts(Intent::Implement);
        red.step_work = true;
        red.verify_failed = true;
        red.grant_verify_credit();
        assert_eq!(red.work_credits, 0, "a failing run proves nothing");

        let mut edited_red = open_plan_facts(Intent::Implement);
        edited_red.mutated = true;
        edited_red.verify_failed = true;
        edited_red.grant_verify_credit();
        assert_eq!(
            edited_red.work_credits, 1,
            "tests after an edit still count"
        );
    }

    #[test]
    fn plan_hints_offer_update_plan_only_when_it_can_succeed() {
        for hint in [
            PLAN_EXECUTE_HINT,
            EDIT_NOW_HINT,
            NEXT_STEP_HINT,
            PLAN_CLOSE_HINT,
            FAILING_TESTS_HINT,
        ] {
            assert!(!hint.contains("Do not call update_plan"), "{hint}");
            assert!(hint.contains("filler"), "{hint}");
        }
        let mut lone = open_plan_facts(Intent::Implement);
        lone.plan_steps = 1;
        assert_eq!(hint_for_facts(&lone, Demand::Edit), EDIT_NOW_HINT);
        assert!(!EDIT_NOW_HINT.contains("update_plan"));
        let mut multi = open_plan_facts(Intent::Implement);
        multi.plan_steps = 2;
        assert_eq!(hint_for_facts(&multi, Demand::Edit), PLAN_EXECUTE_HINT);
        assert!(PLAN_EXECUTE_HINT.contains("call update_plan"));
        multi.verify_credit_granted = true;
        assert_eq!(
            hint_for_facts(&multi, Demand::Edit),
            EDIT_NOW_HINT,
            "this turn's test-run close is spent"
        );
        multi.closed_step = true;
        assert_eq!(hint_for_facts(&multi, Demand::Edit), NEXT_STEP_HINT);
        multi.work_credits = 1;
        assert_eq!(hint_for_facts(&multi, Demand::Edit), PLAN_EXECUTE_HINT);
    }

    /// Tests that passed before the model posted its plan still close a
    /// "Baseline: run cargo test" step, but not a one-step plan.
    #[test]
    fn tests_before_the_plan_count_for_it() {
        let baseline = vec![
            step("Baseline: run cargo test", PlanStatus::Done),
            step(
                "Frame id numbers in frame header/decoder",
                PlanStatus::Active,
            ),
        ];
        let mut facts = TurnFacts::new(Intent::Implement, false);
        facts.ran_verify = true;
        facts.preview_plan(&[], &baseline);
        let accepted = accept_plan(&[], baseline.clone(), &mut facts.work_credits);
        assert_eq!(accepted, baseline);
        assert_eq!(plan_refusal_note(&baseline, &accepted, &facts), None);

        let lone = vec![step("Frame id numbers", PlanStatus::Done)];
        let mut facts = TurnFacts::new(Intent::Implement, false);
        facts.ran_verify = true;
        facts.preview_plan(&[], &lone);
        let accepted = accept_plan(&[], lone.clone(), &mut facts.work_credits);
        assert_eq!(accepted[0].status, PlanStatus::Pending);

        let mut red = TurnFacts::new(Intent::Implement, false);
        red.ran_verify = true;
        red.verify_failed = true;
        red.preview_plan(&[], &baseline);
        assert_eq!(red.work_credits, 0);
    }

    /// After a step closed this turn on tests alone, stopping short of the
    /// next one ends the turn; plan drive picks that step up in a fresh turn.
    /// With edits this turn the rest is still demanded.
    #[test]
    fn a_step_closed_on_tests_alone_ends_the_turn_instead_of_plan_stall() {
        let closed = || {
            let mut facts = open_plan_facts(Intent::Implement);
            facts.plan_steps = 3;
            facts.used_tools = true;
            facts.visible_answer = true;
            facts.closed_step = true;
            facts.verify_credit_granted = true;
            facts
        };
        let mut stopped = closed();
        assert!(is_complete(&decide(Event::ModelStop, &mut stopped)));
        stopped.visible_answer = false;
        assert!(is_complete(&decide(Event::ModelStop, &mut stopped)));

        let mut inspecting = closed();
        let mut decision = Decision::Proceed;
        for _ in 0..20 {
            inspecting.note_round(false, true, false);
            decision = tools_finished(&mut inspecting);
            match decision {
                Decision::Continue { demand, hint } => {
                    assert_eq!(demand, Demand::Edit);
                    assert_eq!(hint, NEXT_STEP_HINT);
                    inspecting.note_continue(demand);
                }
                Decision::Proceed => {}
                _ => break,
            }
        }
        assert!(is_complete(&decision), "{decision:?}");

        let mut unverified = closed();
        unverified.mutated = true;
        unverified.step_work = true;
        let decision = decide(Event::ModelStop, &mut unverified);
        assert!(is_continue(&decision, Demand::Verify), "{decision:?}");

        let mut edited = closed();
        edited.mutated = true;
        edited.ran_verify = true;
        let decision = decide(Event::ModelStop, &mut edited);
        assert!(is_continue(&decision, Demand::Edit), "{decision:?}");

        let mut red = closed();
        red.mutated = true;
        red.step_work = true;
        red.ran_verify = true;
        red.verify_failed = true;
        let mut decision = decide(Event::ModelStop, &mut red);
        while let Decision::Continue { demand, .. } = decision {
            red.note_continue(demand);
            decision = decide(Event::ModelStop, &mut red);
        }
        assert!(is_error(&decision, "plan_stall"), "{decision:?}");
    }

    #[test]
    fn closes_a_step_sees_a_swapped_done_step() {
        let previous = vec![
            step("Baseline: run cargo test", PlanStatus::Done),
            step("Frame id numbers", PlanStatus::Active),
            step("Timing info", PlanStatus::Pending),
        ];
        let replanned = vec![
            step("Frame id numbers", PlanStatus::Done),
            step("Encoder superres", PlanStatus::Active),
        ];
        assert!(closes_a_step(&previous, &replanned));
        assert!(!closes_a_step(&previous, &previous));
        assert!(!closes_a_step(&replanned, &replanned[..1]));
    }

    #[test]
    fn refused_update_plan_names_the_steps_left_open() {
        let previous = vec![
            step("Sequence header", PlanStatus::Done),
            step(
                "Frame id numbers in frame header/decoder",
                PlanStatus::Active,
            ),
            step("Timing info read/write", PlanStatus::Pending),
        ];
        let proposed = vec![
            step("Sequence header", PlanStatus::Done),
            step("Frame id numbers in frame header/decoder", PlanStatus::Done),
            step("Timing info read/write", PlanStatus::Active),
        ];
        let mut credits = 0;
        let accepted = accept_plan(&previous, proposed.clone(), &mut credits);
        let mut facts = open_plan_facts(Intent::Implement);
        facts.plan_steps = previous.len();
        let note = plan_refusal_note(&proposed, &accepted, &facts).expect("refused");
        assert!(note.starts_with("Plan recorded: 1/3 done."), "{note}");
        assert!(
            note.contains("\"Frame id numbers in frame header/decoder\""),
            "{note}"
        );
        assert!(note.contains("run the project's tests"), "{note}");
        facts.verify_credit_granted = true;
        let spent = plan_refusal_note(&proposed, &accepted, &facts).unwrap();
        assert!(!spent.contains("run the project's tests;"), "{spent}");

        let mut credits = 1;
        let funded = accept_plan(&previous, proposed.clone(), &mut credits);
        assert_eq!(plan_refusal_note(&proposed, &funded, &facts), None);
    }

    #[test]
    fn close_active_step_marks_it_done_and_starts_the_next() {
        let plan = vec![
            step("a", PlanStatus::Done),
            step("b", PlanStatus::Active),
            step("c", PlanStatus::Pending),
        ];
        let (next, title) = close_active_step(&plan).unwrap();
        assert_eq!(title, "b");
        let statuses: Vec<_> = next.iter().map(|step| step.status).collect();
        assert_eq!(
            statuses,
            [PlanStatus::Done, PlanStatus::Done, PlanStatus::Active]
        );

        let no_active = vec![
            step("a", PlanStatus::Pending),
            step("b", PlanStatus::Pending),
        ];
        let (next, title) = close_active_step(&no_active).unwrap();
        assert_eq!(title, "a");
        assert_eq!(next[1].status, PlanStatus::Active);

        let last = vec![step("a", PlanStatus::Done), step("b", PlanStatus::Active)];
        let (next, _) = close_active_step(&last).unwrap();
        assert!(!plan_is_open(&next));
        assert_eq!(close_active_step(&next), None);
    }

    #[test]
    fn verify_passed_sees_failures_through_a_pager() {
        use hi_tools::ToolStatus::{Failed, Succeeded};
        assert!(verify_passed(
            Succeeded,
            "test result: ok. 567 passed; 0 failed; 0 ignored"
        ));
        assert!(!verify_passed(
            Succeeded,
            "test result: FAILED. 566 passed; 1 failed; 0 ignored"
        ));
        assert!(!verify_passed(
            Succeeded,
            "error: could not compile `drme` (lib test) due to 2 previous errors"
        ));
        assert!(!verify_passed(Succeeded, "--- FAIL: TestFrameId (0.00s)"));
        assert!(!verify_passed(Failed, "exit status 101"));
    }

    #[test]
    fn closing_the_last_step_satisfies_an_implement_turn_without_an_edit() {
        let mut facts = open_plan_facts(Intent::Implement);
        facts.used_tools = true;
        facts.visible_answer = true;
        facts.ran_verify = true;
        facts.plan_open = false;
        assert!(facts.wants_edit(), "no edit and nothing closed");
        facts.closed_step = true;
        assert!(!facts.wants_edit());
        assert!(is_complete(&decide(Event::ModelStop, &mut facts)));
    }
}
