//! Shared leftover-work drive: classify prompts, decide enqueue vs idle,
//! and judge whether a plan-drive turn made real progress.

use crate::{
    GOAL_CONTINUE_PROMPT, GOAL_DRIVE_STALL_LIMIT, PLAN_DRIVE_PROMPT, PLAN_DRIVE_STALL_LIMIT,
    ProgressEvent, TurnStopReason,
};

/// How a prompt entered the turn loop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DriveKind {
    #[default]
    User,
    Plan,
    Goal,
}

impl DriveKind {
    pub fn from_prompt(prompt: &str) -> Self {
        let trimmed = prompt.trim();
        // A synthetic plan prompt may be persisted or replayed after it is
        // enriched with the active step. Preserve its drive identity in that
        // form; otherwise a replay can accept a text-only no-op as completion.
        if trimmed == PLAN_DRIVE_PROMPT
            || trimmed
                .strip_prefix(PLAN_DRIVE_PROMPT)
                .is_some_and(|rest| rest.starts_with("\nNext:"))
        {
            Self::Plan
        } else if trimmed == crate::GOAL_CONTINUE_PROMPT {
            Self::Goal
        } else {
            Self::User
        }
    }

    pub fn is_drive(self) -> bool {
        !matches!(self, Self::User)
    }
}

/// Why plan auto-drive is not enqueueing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanDriveIdleReason {
    NoLeftover,
    PlanMode,
    Paused,
    Parked,
    GoalDriving,
    Cancelled,
    Infrastructure,
}

/// Whether the next turn should be a synthetic plan-drive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanDriveAction {
    Enqueue,
    Idle { reason: PlanDriveIdleReason },
}

impl PlanDriveAction {
    /// Canonical leftover-work gate. Frontends must not reimplement this.
    pub fn decide(
        plan_incomplete: bool,
        plan_mode: bool,
        paused: bool,
        stall: u32,
        goal_driving: bool,
        stop: Option<TurnStopReason>,
    ) -> Self {
        if plan_mode {
            return Self::Idle {
                reason: PlanDriveIdleReason::PlanMode,
            };
        }
        if goal_driving {
            return Self::Idle {
                reason: PlanDriveIdleReason::GoalDriving,
            };
        }
        match stop {
            // A per-session turn cap is terminal for autonomous driving just
            // like an explicit cancellation. Unlike the per-turn StepLimit,
            // retrying cannot make progress: `turn_count` intentionally stays
            // at the cap, so re-enqueueing would spin forever.
            Some(TurnStopReason::Cancelled | TurnStopReason::TurnLimit) => {
                return Self::Idle {
                    reason: PlanDriveIdleReason::Cancelled,
                };
            }
            Some(TurnStopReason::InfrastructureFailure) => {
                return Self::Idle {
                    reason: PlanDriveIdleReason::Infrastructure,
                };
            }
            _ => {}
        }
        if !plan_incomplete {
            return Self::Idle {
                reason: PlanDriveIdleReason::NoLeftover,
            };
        }
        if paused {
            return Self::Idle {
                reason: PlanDriveIdleReason::Paused,
            };
        }
        if stall >= PLAN_DRIVE_STALL_LIMIT {
            return Self::Idle {
                reason: PlanDriveIdleReason::Parked,
            };
        }
        Self::Enqueue
    }

    pub fn should_enqueue(self) -> bool {
        matches!(self, Self::Enqueue)
    }

    /// Empty Enter / `/plan resume` restart pause and park, not plan-mode.
    pub fn resume_on_empty_enter(self) -> bool {
        matches!(
            self,
            Self::Enqueue
                | Self::Idle {
                    reason: PlanDriveIdleReason::Paused | PlanDriveIdleReason::Parked
                }
        )
    }
}

/// Why leftover-work drive is not enqueueing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriveIdleReason {
    None,
    PlanMode,
    GoalPaused,
    GoalParked,
    PlanPaused,
    PlanParked,
    Cancelled,
    Infrastructure,
}

/// Whether the next turn should be a synthetic goal- or plan-drive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriveAction {
    Enqueue(DriveKind),
    Idle { reason: DriveIdleReason },
}

impl DriveAction {
    pub fn should_enqueue(self) -> bool {
        matches!(self, Self::Enqueue(DriveKind::Plan | DriveKind::Goal))
    }

    pub fn prompt(self) -> Option<&'static str> {
        match self {
            Self::Enqueue(DriveKind::Goal) => Some(GOAL_CONTINUE_PROMPT),
            Self::Enqueue(DriveKind::Plan) => Some(PLAN_DRIVE_PROMPT),
            _ => None,
        }
    }

    /// Empty Enter / resume restart pause and park, not plan-mode.
    pub fn resume_on_empty_enter(self) -> bool {
        matches!(
            self,
            Self::Enqueue(DriveKind::Plan | DriveKind::Goal)
                | Self::Idle {
                    reason: DriveIdleReason::GoalPaused
                        | DriveIdleReason::GoalParked
                        | DriveIdleReason::PlanPaused
                        | DriveIdleReason::PlanParked
                }
        )
    }

    /// Synthetic prompt empty Enter / `/goal resume` / `/plan resume` should submit.
    pub fn resume_prompt(self) -> Option<&'static str> {
        if !self.resume_on_empty_enter() {
            return None;
        }
        match self {
            Self::Enqueue(DriveKind::Goal)
            | Self::Idle {
                reason: DriveIdleReason::GoalPaused | DriveIdleReason::GoalParked,
            } => Some(GOAL_CONTINUE_PROMPT),
            Self::Enqueue(DriveKind::Plan)
            | Self::Idle {
                reason: DriveIdleReason::PlanPaused | DriveIdleReason::PlanParked,
            } => Some(PLAN_DRIVE_PROMPT),
            _ => None,
        }
    }

    pub fn from_plan(plan: PlanDriveAction) -> Self {
        match plan {
            PlanDriveAction::Enqueue => Self::Enqueue(DriveKind::Plan),
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::PlanMode,
            } => Self::Idle {
                reason: DriveIdleReason::PlanMode,
            },
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::Paused,
            } => Self::Idle {
                reason: DriveIdleReason::PlanPaused,
            },
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::Parked,
            } => Self::Idle {
                reason: DriveIdleReason::PlanParked,
            },
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::Cancelled,
            } => Self::Idle {
                reason: DriveIdleReason::Cancelled,
            },
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::Infrastructure,
            } => Self::Idle {
                reason: DriveIdleReason::Infrastructure,
            },
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::NoLeftover | PlanDriveIdleReason::GoalDriving,
            } => Self::Idle {
                reason: DriveIdleReason::None,
            },
        }
    }
}

/// Consecutive synthetic drive turns in one-shot before the loop stops.
pub const ONE_SHOT_DRIVE_TURN_LIMIT: u32 = 32;

/// Live goal-drive status for `/goal status` and report JSON.
pub fn goal_drive_status(goal_leftover: bool, paused: bool, stall: u32) -> &'static str {
    if !goal_leftover {
        "off"
    } else if paused {
        "paused"
    } else if stall >= GOAL_DRIVE_STALL_LIMIT {
        "parked"
    } else {
        "running"
    }
}

/// Park copy when goal-drive stalls out. Never suggests `/retry`.
pub fn goal_drive_park_message(leftover: Option<&str>) -> String {
    format!(
        "goal drive parked: {}",
        leftover.unwrap_or("unfinished goal")
    )
}

/// Copy when a stuck goal step is skipped so the rest of the plan can drive.
pub fn goal_drive_skip_message(failed: &str, next: Option<&str>) -> String {
    match next {
        Some(next) => format!("goal step stalled — skipped `{failed}`; driving `{next}`"),
        None => format!("goal step stalled — skipped `{failed}`"),
    }
}

/// Copy when stall-skipped steps are queued for another pass.
pub fn goal_drive_requeue_message(count: usize) -> String {
    format!("goal drive: {count} stalled step(s) queued for a second pass")
}

/// Result of recording whether a goal-drive turn made progress.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoalDriveProgress {
    /// Stall counter unchanged (already zero on progress, or no-op).
    Unchanged,
    /// Stall counter reset after real progress.
    Reset,
    /// No progress; stall is still below the park/skip limit.
    Stalled { stall: u32 },
    /// The stuck step was skipped; the next pending step is now active.
    Skipped {
        failed: String,
        next: Option<String>,
    },
    /// Stall-skipped Failed steps were returned to Pending for a second pass.
    Requeued { count: usize },
    /// Parked: thrashing, nothing left to skip to, or a single stuck step.
    Parked,
}

/// Live plan-drive status for `/plan status` and report JSON.
pub fn plan_drive_status(plan_incomplete: bool, paused: bool, stall: u32) -> &'static str {
    if !plan_incomplete {
        "off"
    } else if paused {
        "paused"
    } else if stall >= PLAN_DRIVE_STALL_LIMIT {
        "parked"
    } else {
        "running"
    }
}

const PLAN_DRIVE_PROGRESS_REASONS: &[&str] = &[
    "changed plan state",
    "substantive edit",
    "successful mutation",
    "successful validation after mutation",
];

/// Whether a goal-drive turn made real progress: the structured goal changed
/// in a way that counts as drive work, or files were edited. File edits must
/// count even when the model never called `update_plan` — otherwise a
/// productive cap/resume turn parks on the next "already done" text answer.
pub fn goal_drive_made_progress(
    before: Option<&crate::Goal>,
    after: Option<&crate::Goal>,
    changed_files: &[String],
) -> bool {
    if !changed_files.is_empty() {
        return true;
    }
    match (after, before) {
        (Some(after), Some(before)) => after.drive_state_changed_since(before),
        _ => true,
    }
}

/// Whether a plan-drive turn made real progress: the next step identity
/// changed, a mutation landed, or a plan-state meaningful event fired.
/// Reads classified as meaningful (`new file evidence`, `new targeted search
/// evidence`) do not count.
pub fn plan_drive_made_progress(
    before_step: Option<&str>,
    after_step: Option<&str>,
    telemetry: &[ProgressEvent],
    changed_files: &[String],
) -> bool {
    if before_step != after_step {
        return true;
    }
    if !changed_files.is_empty() {
        return true;
    }
    telemetry.iter().any(|event| {
        event.kind == "meaningful" && PLAN_DRIVE_PROGRESS_REASONS.contains(&event.reason.as_str())
    })
}

/// Consecutive no-progress plan-drive turns. Any user turn or a progress
/// drive resets the count.
pub fn next_plan_drive_stall(was_driving: bool, made_progress: bool, current: u32) -> u32 {
    if was_driving && !made_progress {
        current.saturating_add(1)
    } else {
        0
    }
}

/// Park copy when plan-drive stalls out. Names leftover work; never suggests `/retry`.
pub fn plan_drive_park_message(leftover: Option<&str>) -> String {
    format!(
        "plan drive parked: {}",
        leftover.unwrap_or("unfinished plan")
    )
}

/// Transcript chrome for a synthetic drive prompt. `None` when `prompt` is a
/// normal user message — callers should echo that message instead.
pub fn drive_chrome_line(
    prompt: &str,
    next_plan_step: Option<&str>,
    next_goal: Option<&str>,
) -> Option<String> {
    match DriveKind::from_prompt(prompt) {
        DriveKind::Plan => Some(match next_plan_step {
            Some(step) => format!("⟳ plan drive — {step}"),
            None => "⟳ plan drive".into(),
        }),
        DriveKind::Goal => Some(match next_goal {
            Some(goal) => format!("⟳ goal drive — {goal}"),
            None => "⟳ goal drive".into(),
        }),
        DriveKind::User => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProgressEvent;

    #[test]
    fn drive_kind_classifies_synthetic_prompts() {
        assert_eq!(DriveKind::from_prompt(PLAN_DRIVE_PROMPT), DriveKind::Plan);
        assert_eq!(
            DriveKind::from_prompt(&format!(
                "{PLAN_DRIVE_PROMPT}\nNext: Verify with cargo check/test. Use your tools now."
            )),
            DriveKind::Plan
        );
        assert_eq!(
            DriveKind::from_prompt(crate::GOAL_CONTINUE_PROMPT),
            DriveKind::Goal
        );
        assert_eq!(DriveKind::from_prompt("fix the bug"), DriveKind::User);
        assert_eq!(
            DriveKind::from_prompt(&format!("{PLAN_DRIVE_PROMPT} please")),
            DriveKind::User,
            "ordinary user text that merely starts similarly is not synthetic"
        );
    }

    #[test]
    fn decide_covers_idle_reasons() {
        let enqueue = PlanDriveAction::decide(true, false, false, 0, false, None);
        assert_eq!(enqueue, PlanDriveAction::Enqueue);
        assert!(enqueue.should_enqueue());
        assert!(enqueue.resume_on_empty_enter());

        assert_eq!(
            PlanDriveAction::decide(true, true, false, 0, false, None),
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::PlanMode
            }
        );
        assert_eq!(
            PlanDriveAction::decide(true, false, false, 0, true, None),
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::GoalDriving
            }
        );
        assert_eq!(
            PlanDriveAction::decide(
                true,
                false,
                false,
                0,
                false,
                Some(TurnStopReason::Cancelled)
            ),
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::Cancelled
            }
        );
        assert_eq!(
            PlanDriveAction::decide(
                true,
                false,
                false,
                0,
                false,
                Some(TurnStopReason::TurnLimit)
            ),
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::Cancelled
            },
            "a session turn cap must not requeue unfinished plan work"
        );
        assert_eq!(
            PlanDriveAction::decide(false, false, false, 0, false, None),
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::NoLeftover
            }
        );
        let paused = PlanDriveAction::decide(true, false, true, 0, false, None);
        assert_eq!(
            paused,
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::Paused
            }
        );
        assert!(paused.resume_on_empty_enter());
        let parked =
            PlanDriveAction::decide(true, false, false, PLAN_DRIVE_STALL_LIMIT, false, None);
        assert_eq!(
            parked,
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::Parked
            }
        );
        assert!(parked.resume_on_empty_enter());
        assert!(
            !PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::PlanMode
            }
            .resume_on_empty_enter()
        );
    }

    #[test]
    fn drive_action_goal_wins_and_resume_covers_pause_park() {
        let goal = DriveAction::Enqueue(DriveKind::Goal);
        assert_eq!(goal.prompt(), Some(GOAL_CONTINUE_PROMPT));
        assert!(goal.resume_on_empty_enter());
        let paused = DriveAction::Idle {
            reason: DriveIdleReason::GoalPaused,
        };
        assert!(paused.resume_on_empty_enter());
        assert!(
            DriveAction::Idle {
                reason: DriveIdleReason::GoalParked
            }
            .resume_on_empty_enter()
        );
        assert!(
            !DriveAction::Idle {
                reason: DriveIdleReason::PlanMode
            }
            .resume_on_empty_enter()
        );
        assert_eq!(
            DriveAction::from_plan(PlanDriveAction::Enqueue),
            DriveAction::Enqueue(DriveKind::Plan)
        );
    }

    #[test]
    fn plan_drive_progress_ignores_read_evidence() {
        let read = [ProgressEvent {
            kind: "meaningful".into(),
            reason: "new file evidence".into(),
            signature: None,
        }];
        let edit = [ProgressEvent {
            kind: "meaningful".into(),
            reason: "substantive edit".into(),
            signature: None,
        }];
        assert!(!plan_drive_made_progress(
            Some("a"),
            Some("a"),
            &read,
            &[] as &[String]
        ));
        assert!(plan_drive_made_progress(
            Some("a"),
            Some("a"),
            &edit,
            &[] as &[String]
        ));
        assert!(plan_drive_made_progress(
            Some("a"),
            Some("b"),
            &[],
            &[] as &[String]
        ));
        assert!(plan_drive_made_progress(
            Some("a"),
            Some("a"),
            &[],
            &["src/lib.rs".into()]
        ));
        let goal = crate::Goal::new("ship", vec!["a".into(), "b".into()]);
        assert!(
            goal_drive_made_progress(Some(&goal), Some(&goal), &["crates/api/src/lib.rs".into()]),
            "file edits count as goal-drive progress even when the goal state is unchanged"
        );
        assert!(
            !goal_drive_made_progress(Some(&goal), Some(&goal), &[]),
            "an unchanged goal with no file edits is a stall"
        );
        assert_eq!(next_plan_drive_stall(true, false, 3), 4);
        let parked = plan_drive_park_message(Some("1/2 remaining — wire the scheduler"));
        assert!(parked.contains("1/2 remaining — wire the scheduler"));
        assert!(!parked.contains("/retry"));
    }

    #[test]
    fn drive_chrome_hides_synthetic_prompt() {
        let line = drive_chrome_line(PLAN_DRIVE_PROMPT, Some("wire the scheduler"), None)
            .expect("plan chrome");
        assert_eq!(line, "⟳ plan drive — wire the scheduler");
        assert!(!line.contains(PLAN_DRIVE_PROMPT));
        let goal = drive_chrome_line(crate::GOAL_CONTINUE_PROMPT, None, Some("ship it"))
            .expect("goal chrome");
        assert_eq!(goal, "⟳ goal drive — ship it");
        assert!(drive_chrome_line("fix the bug", Some("x"), None).is_none());
    }
}
