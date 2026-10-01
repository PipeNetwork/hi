//! Keep an open plan moving without the user retyping "Continue the open plan."
//!
//! Every live drme stall ended the same way: a turn stopped (plan_stall, a
//! tool storm, a request error, or a plain "Next: …" summary) with plan steps
//! left, and hi sat idle until the user typed the continue prompt. After each
//! turn the drive starts that prompt itself, and pauses when turns stop making
//! progress, the user interrupts, or the key is rejected.

use crate::session::PlanDrive;
use crate::{CONTINUE_PLAN_PROMPT, Harness, TurnOutcome, TurnStopReason, completion};
use hi_tools::{PlanStatus, PlanStep};

/// Turns in a row with no file change and no finished step before pausing.
pub const MAX_DRIVE_STALLS: u32 = 2;
/// Autonomous turns in a row before pausing, whatever they achieve.
pub const DEFAULT_MAX_DRIVE_TURNS: u32 = 30;

/// Plan state captured before a turn so the drive can tell whether it moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DriveMark {
    plan_done: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DriveStep {
    /// Start another turn with this prompt.
    Continue(String),
    /// The drive stopped itself; tell the user why.
    Paused(String),
    /// Nothing to drive: the plan is closed, the drive is off or already paused.
    Idle,
}

/// Prompt the drive sends so a recap-and-question cop-out cannot count as
/// finishing the turn. Always starts with [`CONTINUE_PLAN_PROMPT`] so
/// [`completion::is_continuation_prompt`] still classifies it as Implement.
/// When a passing test run could close the step it also names that route;
/// "edit or write" alone pushes the model into filler edits on a step that
/// is already implemented. `step_edited` is [`PlanDrive::step_edited`].
pub fn drive_continue_prompt(plan: &[PlanStep], stall: u32, step_edited: bool) -> String {
    let step = plan
        .iter()
        .find(|step| step.status == PlanStatus::Active)
        .or_else(|| plan.iter().find(|step| step.status != PlanStatus::Done))
        .map(|step| step.title.as_str())
        .unwrap_or("the next unfinished step");
    let close = if completion::tests_may_close_plan_step(plan, step_edited) {
        " If the current tree already implements it, run the project's tests and call update_plan to mark it done, then start the next step."
    } else {
        ""
    };
    if stall == 0 {
        format!(
            "{CONTINUE_PLAN_PROMPT} Implement this step now with edit or write: {step}.{close} Do not recap status, ask what to do next, or make filler edits."
        )
    } else {
        format!(
            "{CONTINUE_PLAN_PROMPT} The previous turn recapped and changed nothing. Implement this step now with edit or write: {step}.{close} Do not repeat the last summary or make filler edits."
        )
    }
}

fn done_steps(plan: &[PlanStep]) -> usize {
    plan.iter()
        .filter(|step| step.status == PlanStatus::Done)
        .count()
}

/// `HI_PLAN_DRIVE=0` (or `off`/`false`/`no`) turns the drive off.
fn drive_enabled() -> bool {
    !matches!(
        std::env::var("HI_PLAN_DRIVE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "off" | "false" | "no"
    )
}

fn max_drive_turns() -> u32 {
    std::env::var("HI_PLAN_DRIVE_MAX_TURNS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(DEFAULT_MAX_DRIVE_TURNS)
}

fn credential_error(message: &str) -> bool {
    message.contains("pipe http 401") || message.contains("pipe http 403")
}

impl Harness {
    pub fn plan_drive_mark(&self) -> DriveMark {
        DriveMark {
            plan_done: done_steps(&self.plan),
        }
    }

    /// Plan progress as `(done, total)` for status lines.
    pub fn plan_progress(&self) -> (usize, usize) {
        (done_steps(&self.plan), self.plan.len())
    }

    /// Decide what follows a finished turn. `driven` counts autonomous turns
    /// already started for the current user prompt.
    pub fn plan_drive_after_turn(
        &mut self,
        mark: DriveMark,
        outcome: Option<&TurnOutcome>,
        driven: u32,
    ) -> DriveStep {
        if !completion::plan_is_open(&self.plan) {
            if self.plan_drive.stall != 0 || self.plan_drive.step_edited {
                self.plan_drive.stall = 0;
                self.plan_drive.step_edited = false;
                self.persist_plan_drive();
            }
            return DriveStep::Idle;
        }
        if !drive_enabled()
            || self.plan_drive.paused
            || self.review_drive.is_active()
            || self.ladder.is_active()
        {
            return DriveStep::Idle;
        }
        let Some(outcome) = outcome else {
            return self.pause_plan_drive("the last turn could not run");
        };
        if outcome.stop_reason == TurnStopReason::Cancelled {
            return self.pause_plan_drive("the turn was interrupted");
        }
        if outcome.error.as_deref().is_some_and(credential_error) {
            return self.pause_plan_drive("Pipe rejected the API key");
        }
        let progressed = !outcome.changed_files.is_empty()
            || done_steps(&self.plan) > mark.plan_done
            || outcome.attempted_mutation;
        self.plan_drive.stall = if progressed {
            0
        } else {
            self.plan_drive.stall.saturating_add(1)
        };
        if self.plan_drive.stall >= MAX_DRIVE_STALLS {
            let stalls = self.plan_drive.stall;
            return self.pause_plan_drive(&format!(
                "{stalls} turns in a row changed no files and finished no plan step"
            ));
        }
        let cap = max_drive_turns();
        if driven >= cap {
            return self.pause_plan_drive(&format!("ran {cap} plan turns in a row"));
        }
        self.persist_plan_drive();
        DriveStep::Continue(drive_continue_prompt(
            &self.plan,
            self.plan_drive.stall,
            self.plan_drive.step_edited,
        ))
    }

    /// A prompt the user typed resumes a drive that paused itself.
    pub fn resume_plan_drive_on_input(&mut self) {
        if self.plan_drive.paused && self.plan_drive.resume_on_user_input {
            self.plan_drive = PlanDrive {
                step_edited: self.plan_drive.step_edited,
                ..PlanDrive::default()
            };
            self.persist_plan_drive();
        }
    }

    /// Record whether the active step has an applied edit. Cleared when the
    /// step closes or a new plan starts.
    pub(crate) fn set_step_edited(&mut self, edited: bool) {
        if self.plan_drive.step_edited != edited {
            self.plan_drive.step_edited = edited;
            self.persist_plan_drive();
        }
    }

    fn pause_plan_drive(&mut self, reason: &str) -> DriveStep {
        self.plan_drive.paused = true;
        self.plan_drive.resume_on_user_input = true;
        self.persist_plan_drive();
        DriveStep::Paused(reason.to_string())
    }

    fn persist_plan_drive(&mut self) {
        if let Some(session) = &mut self.session {
            let _ = session.record_plan_drive(&self.plan_drive);
        }
    }
}
