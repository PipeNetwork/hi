//! Harness-owned turn completion policy.
//!
//! The turn loop asks the model for the next action; this module decides
//! whether that response may finish the turn. Hints are the voice of a
//! demand, not a second stop machine. Progress is facts: mutation, verify,
//! an open plan, a user-visible answer, consecutive stall rounds.

use hi_ai::{Content, Message};
use hi_tools::{PlanStatus, PlanStep, is_coordination, is_inspect_tool, plan_steps_from_arguments};

/// Consecutive stall-only tool rounds before the harness demands Edit / Verify / Verdict.
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
Do not repeat the same grep/read/list. If a search returned no matches, \
the code is not in the tree — add it with edit/write. Continue the task \
or write the complete reply to the user. Do not stop without that reply.";

pub const UNVERIFIED_FIX_HINT: &str = "\
You were asked to review and fix. Run `cargo test` (or the project's test \
command) now to verify the current tree. If tests fail, fix them with `edit`. \
If they pass, give a short verdict and stop. Do not keep grepping.";

#[allow(dead_code)]
pub const NEXT_ACTION_EDIT_HINT: &str = "\
You were asked to review and fix. Stop inspecting and call `edit`/`write` on \
the failing code, or run `cargo test` if you do not yet know what fails.";

pub const PLAN_EXECUTE_HINT: &str = "\
Stop inspecting. Your next action MUST be `edit` or `write` on the files that \
need to change. If a plan is open, implement the active step now. Do not mark \
a step done unless you changed the tree. Do not start a new grep or list. \
If you were paging a file, you have enough — edit now. Do not call \
update_plan. Call edit/write in this response.";

pub const INSPECT_STOP_VERDICT_HINT: &str = "\
Stop inspecting. Write a short user-visible answer now: what to change, or \
that you will not change anything. Do not grep or read the same files again.";

const EMPTY_STOP_MSG: &str = "model stopped after tool work with no user-visible answer";
const INSPECT_STOP_MSG: &str = "stopped inspecting without applying the requested changes";
const PLAN_STALL_MSG: &str = "plan still has unfinished steps but the turn stopped without an edit";

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
        }
    }

    pub fn grant_mutation_credit(&mut self) {
        self.work_credits = self.work_credits.saturating_add(1);
    }

    pub fn grant_verify_credit(&mut self) {
        if self.mutated && !self.verify_credit_granted {
            self.work_credits = self.work_credits.saturating_add(1);
            self.verify_credit_granted = true;
        }
    }

    /// Remaining plan steps on Implement are still an edit demand after a
    /// partial patch. Verify wins if tests have not run since the last edit.
    pub fn wants_edit(&self) -> bool {
        if self.wants_verify() {
            return false;
        }
        match self.intent {
            Intent::Implement => !self.mutated || self.plan_open,
            Intent::Fix => self.plan_open && !self.mutated,
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
    /// Mutation always resets. Stall-only rounds increment. `cargo test`
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
    if !facts.used_tools {
        return Decision::Complete;
    }
    if !facts.visible_answer {
        if facts.wants_verify() && can_continue(facts, Demand::Verify) {
            return continue_demand(Demand::Verify);
        }
        if facts.wants_edit() && can_continue(facts, Demand::Edit) {
            return continue_demand(Demand::Edit);
        }
        if can_continue(facts, Demand::VisibleAnswer) {
            return continue_demand(Demand::VisibleAnswer);
        }
        return Decision::Error {
            kind: "empty_stop",
            message: EMPTY_STOP_MSG,
        };
    }
    if facts.wants_verify() {
        if can_continue(facts, Demand::Verify) {
            return continue_demand(Demand::Verify);
        }
        return unmet_error(facts);
    }
    if facts.wants_edit() {
        if can_continue(facts, Demand::Edit) {
            return continue_demand(Demand::Edit);
        }
        if facts.plan_open && !facts.last_chance {
            facts.last_chance = true;
            return continue_demand(Demand::Edit);
        }
        return unmet_error(facts);
    }
    Decision::Complete
}

fn decide_tools_finished(facts: &mut TurnFacts, inspect_repeat_capped: bool) -> Decision {
    if inspect_repeat_capped {
        return capped_repeat_demand(facts);
    }
    let pressure = facts.stall_pressure();
    if pressure < STALL_ROUNDS_BEFORE_DEMAND {
        return Decision::Proceed;
    }
    let demand = stall_demand(facts);
    if pressure < STALL_ROUNDS_BEFORE_ERROR {
        return continue_demand(demand);
    }
    // The inspect that hit the cap often *is* the last file the model
    // needed (live ~/chat: plan + paginated server/db/state, then ws.rs).
    // Give one edit demand with that result in context before plan_stall.
    if facts.wants_edit() && facts.plan_open && !facts.last_chance {
        facts.last_chance = true;
        return continue_demand(demand);
    }
    unmet_error(facts)
}

fn capped_repeat_demand(facts: &mut TurnFacts) -> Decision {
    let demand = stall_demand(facts);
    if can_continue(facts, demand) {
        return continue_demand(demand);
    }
    if !facts.last_chance {
        facts.last_chance = true;
        return continue_demand(demand);
    }
    unmet_error(facts)
}

fn unmet_error(facts: &TurnFacts) -> Decision {
    if facts.wants_edit() && facts.plan_open {
        return Decision::Error {
            kind: "plan_stall",
            message: PLAN_STALL_MSG,
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

fn continue_demand(demand: Demand) -> Decision {
    Decision::Continue {
        demand,
        hint: hint_for(demand),
    }
}

pub fn hint_for(demand: Demand) -> &'static str {
    match demand {
        Demand::Edit => PLAN_EXECUTE_HINT,
        Demand::Verify => UNVERIFIED_FIX_HINT,
        Demand::Verdict => INSPECT_STOP_VERDICT_HINT,
        Demand::VisibleAnswer => EMPTY_AFTER_TOOLS_HINT,
    }
}

pub fn status_for(demand: Demand) -> &'static str {
    match demand {
        Demand::Edit => "task in progress; asking for an edit",
        Demand::Verify => "review-and-fix stopped after inspection; asking for tests",
        Demand::Verdict => "inspect-repeat stop; asking for a verdict",
        Demand::VisibleAnswer => "empty model stop after tools; continuing",
    }
}

/// TypeSafe may only change Edit/Verify/Verdict flavor. `Inspect` is ignored.
///
/// An open plan that has not been edited yet cannot become tests or a
/// verdict: that is the live ~/chat "lets build all of that" stall (cargo
/// test of the current tree, inspect-repeat, then `plan_stall`). Remaining
/// plan steps are also not a cop-out after cargo test.
pub fn flavor_demand(current: Demand, typesafe: crate::NextAction, facts: &TurnFacts) -> Demand {
    if current == Demand::VisibleAnswer || typesafe == crate::NextAction::Inspect {
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
        "make those changes",
        "apply those",
        "implement that",
        "implement those",
        "implement it",
    ];
    if PHRASES.iter().any(|phrase| lower.contains(phrase)) {
        return true;
    }
    lower
        .split(|ch: char| !ch.is_ascii_alphabetic())
        .any(|word| matches!(word, "implement" | "implements"))
}

pub fn plan_is_open(plan: &[PlanStep]) -> bool {
    !plan.is_empty() && !PlanStep::all_complete(plan)
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
        if step.status != PlanStatus::Done {
            continue;
        }
        let was_done = previous
            .iter()
            .any(|prior| prior.title == step.title && prior.status == PlanStatus::Done);
        if was_done {
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

pub fn looks_like_verify(name: &str, arguments: &str) -> bool {
    if !name.eq_ignore_ascii_case("bash") {
        return false;
    }
    let lower = bash_command(arguments).to_ascii_lowercase();
    lower.contains("cargo test")
        || lower.contains("cargo check")
        || lower.contains("cargo clippy")
        || lower.contains("npm test")
        || lower.contains("npx vitest")
        || lower.contains("pytest")
        || lower.contains("-m unittest")
        || lower.contains("go test")
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
        let demand = tools_finished(&mut facts);
        assert!(is_continue(&demand, Demand::Verdict), "{demand:?}");
        assert!(!is_continue(&demand, Demand::Edit));
        facts.note_continue(Demand::Verdict);
        assert!(is_complete(&decide(Event::ModelStop, &mut facts)));
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
}
