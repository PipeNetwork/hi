//! Shared leftover-work drive: classify prompts, decide enqueue vs idle,
//! and judge whether a plan-drive turn made real progress.

use std::collections::HashSet;

use crate::{
    GOAL_CONTINUE_PROMPT, GOAL_DRIVE_STALL_LIMIT, PLAN_DRIVE_PROMPT, PLAN_DRIVE_STALL_LIMIT,
    ProgressEvent, TurnStopReason,
};
use sha2::{Digest, Sha256};

const DRIVE_EVIDENCE_REASONS: &[&str] = &["new file evidence", "new targeted search evidence"];

/// Exact evidence already credited inside one structural drive scope.
///
/// Only fixed-size SHA-256 values are retained: raw read paths/search patterns
/// never enter session metadata. The set deliberately does not evict entries.
/// Pure investigation steps may keep crediting novel evidence, while cycling
/// through any finite collection eventually stays non-novel. The lifecycle
/// layer separately limits evidence-only orientation on implementation steps.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DriveEvidenceLedger {
    hashes: HashSet<String>,
}

impl DriveEvidenceLedger {
    pub(crate) fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    pub(crate) fn restore(&mut self, hashes: impl IntoIterator<Item = String>) {
        self.hashes = hashes
            .into_iter()
            .filter(|hash| valid_drive_evidence_hash(hash))
            .collect();
    }

    pub(crate) fn snapshot(&self) -> Vec<String> {
        let mut hashes = self.hashes.iter().cloned().collect::<Vec<_>>();
        hashes.sort_unstable();
        hashes
    }

    pub(crate) fn clear(&mut self) -> bool {
        if self.hashes.is_empty() {
            return false;
        }
        self.hashes.clear();
        true
    }

    /// Record every signed evidence event and return only hashes not previously
    /// credited in this structural scope.
    pub(crate) fn record_novel(&mut self, telemetry: &crate::TurnTelemetry) -> Vec<String> {
        let mut added = Vec::new();
        // This complete, non-diagnostic collection is not subject to progress
        // trail head/tail compaction.
        for hash in &telemetry.drive_evidence_hashes {
            if valid_drive_evidence_hash(hash) && self.hashes.insert(hash.clone()) {
                added.push(hash.clone());
            }
        }
        // Compatibility/focused-test fallback for telemetry built before the
        // dedicated complete collection was populated.
        for event in &telemetry.progress_events {
            if !progress_event_is_drive_evidence(event) {
                continue;
            }
            let Some(signature) = event.signature.as_deref() else {
                // A missing signature cannot prove cross-turn novelty. Valid
                // production read/search calls always carry one; malformed
                // calls already classify as errors rather than evidence.
                continue;
            };
            let hash = hash_drive_evidence_signature(signature);
            if self.hashes.insert(hash.clone()) {
                added.push(hash);
            }
        }
        added
    }
}

pub(crate) fn hash_drive_evidence_signature(signature: &str) -> String {
    format!("{:x}", Sha256::digest(signature.as_bytes()))
}

fn valid_drive_evidence_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn progress_event_is_drive_evidence(event: &ProgressEvent) -> bool {
    event.kind == "meaningful" && DRIVE_EVIDENCE_REASONS.contains(&event.reason.as_str())
}

pub(crate) fn progress_event_is_structural_or_mutation(event: &ProgressEvent) -> bool {
    event.kind == "meaningful"
        && matches!(
            event.reason.as_str(),
            "changed plan state"
                | "substantive edit"
                | "successful mutation"
                | "successful validation after mutation"
        )
}

/// Whether read/search novelty may keep the current plan step running without
/// any structural or workspace progress.
///
/// Pure investigation and validation steps can legitimately finish a turn
/// with evidence alone. Implementation steps cannot: they get one orientation
/// turn, tracked by the persisted evidence ledger, and subsequent read-only
/// turns advance the semantic stall circuit even when they inspect new files.
pub(crate) fn plan_step_allows_continual_evidence(step: Option<&str>) -> bool {
    step.is_some_and(crate::agent::plan_goal::is_meta_milestone)
}

/// Step-aware instruction for a fresh plan-recovery epoch.
///
/// The recovery window retains only evidence fingerprints, not raw tool
/// output. Say that explicitly so the model may re-read the narrow facts it
/// needs rather than being ordered to act on context that no longer exists.
pub(crate) fn plan_recovery_instruction(step: Option<&str>) -> &'static str {
    if plan_step_allows_continual_evidence(step) {
        "The preceding automatic turn did not advance this investigation or validation step. This fresh strategy epoch dropped prior tool-output contents while retaining their fingerprints for repetition detection. Use a narrower question or command, gather genuinely new targeted evidence when needed, then record a concrete conclusion or validation result and advance the plan. Re-read old evidence only when necessary; cycling through inspection alone will not count as progress."
    } else {
        "The preceding automatic turn only inspected and did not advance this implementation step. This fresh strategy epoch dropped prior tool-output contents while retaining their fingerprints for repetition detection. Re-read only the narrow evidence needed to act, then make the smallest safe mutation and validate it, or use a genuinely different implementation/delegation strategy. Do not reopen broad inspection."
    }
}

/// How a prompt entered the turn loop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DriveKind {
    #[default]
    User,
    Plan,
    Goal,
}

/// Why checklist auto-drive is paused.
///
/// A manual pause is durable user intent and survives ordinary conversation.
/// An interruption pause is a stop latch: it prevents an abandoned synthetic
/// turn from restarting by itself, but the next real user turn consumes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PlanDrivePause {
    #[default]
    Running,
    Manual,
    Interrupted,
}

impl PlanDrivePause {
    pub(crate) fn is_paused(self) -> bool {
        !matches!(self, Self::Running)
    }

    pub(crate) fn resumes_on_user_input(self) -> bool {
        matches!(self, Self::Interrupted)
    }
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

/// Whether a settled outcome must stop autonomous plan/goal driving until the
/// user explicitly intervenes.
///
/// A landed mutation can be useful even when verification is unavailable, and
/// explicitly configured per-turn caps intentionally split work across turns.
/// Those outcomes remain resumable. Actual verification/review failures and
/// every cancellation, block, infrastructure failure, or no-progress stop are
/// fail-closed.
pub(crate) fn outcome_blocks_automatic_drive(outcome: &crate::TurnOutcome) -> bool {
    if matches!(
        outcome.status,
        crate::TurnStatus::Cancelled | crate::TurnStatus::Blocked
    ) || matches!(
        outcome.stop_reason,
        TurnStopReason::Cancelled
            | TurnStopReason::TurnLimit
            | TurnStopReason::InfrastructureFailure
            | TurnStopReason::NoProgress
    ) {
        return true;
    }
    if outcome.status != crate::TurnStatus::Failed {
        return false;
    }
    match outcome.stop_reason {
        TurnStopReason::VerificationUnavailable => outcome.changed_files.is_empty(),
        TurnStopReason::StepLimit | TurnStopReason::ToolLimit | TurnStopReason::TimeLimit => false,
        _ => true,
    }
}

/// Why plan auto-drive is not enqueueing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanDriveIdleReason {
    NoLeftover,
    PlanMode,
    ApprovalParked,
    Paused,
    Parked,
    GoalDriving,
    Cancelled,
    Blocked,
    NoProgress,
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
            Some(TurnStopReason::NoProgress) => {
                return Self::Idle {
                    reason: PlanDriveIdleReason::NoProgress,
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
    PlanApprovalParked,
    GoalPaused,
    GoalParked,
    PlanPaused,
    PlanParked,
    Cancelled,
    Blocked,
    NoProgress,
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
                reason: PlanDriveIdleReason::ApprovalParked,
            } => Self::Idle {
                reason: DriveIdleReason::PlanApprovalParked,
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
                reason: PlanDriveIdleReason::Blocked,
            } => Self::Idle {
                reason: DriveIdleReason::Blocked,
            },
            PlanDriveAction::Idle {
                reason: PlanDriveIdleReason::NoProgress,
            } => Self::Idle {
                reason: DriveIdleReason::NoProgress,
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

/// Unlimited sentinel for one-shot synthetic drive turns. Fault/no-progress
/// guards still park a stuck drive; productive work has no default turn count.
pub const ONE_SHOT_DRIVE_TURN_LIMIT: u32 = u32::MAX;

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

/// Park copy when goal-drive runs out of useful progress. Never suggests `/retry`.
pub fn goal_drive_park_message(leftover: Option<&str>) -> String {
    format!(
        "goal drive parked: {}",
        leftover.unwrap_or("unfinished goal")
    )
}

/// Copy when a stuck goal step is skipped so the rest of the plan can drive.
pub fn goal_drive_skip_message(failed: &str, next: Option<&str>) -> String {
    match next {
        Some(next) => format!("goal step made no progress — skipped `{failed}`; driving `{next}`"),
        None => format!("goal step made no progress — skipped `{failed}`"),
    }
}

/// Copy when no-progress steps are queued for another pass.
pub fn goal_drive_requeue_message(count: usize) -> String {
    format!("goal drive: {count} no-progress step(s) queued for a second pass")
}

/// Result of recording whether a goal-drive turn made progress.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoalDriveProgress {
    /// Stall counter unchanged (already zero on progress, or no-op).
    Unchanged,
    /// Stall counter reset after real progress.
    Reset,
    /// No progress; the retry count is still below the park/skip limit.
    NoProgress { count: u32 },
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
    // Investigation is productive work too. These labels are novel inside one
    // turn; Agent's drive evidence ledger enforces novelty across turns.
    "new file evidence",
    "new targeted search evidence",
];

pub(crate) fn progress_event_counts_as_plan_drive(event: &ProgressEvent) -> bool {
    event.kind == "meaningful" && PLAN_DRIVE_PROGRESS_REASONS.contains(&event.reason.as_str())
}

/// Turn-local goal progress classifier. Multi-turn frontends must use
/// [`crate::Agent::goal_drive_turn_made_progress`] so read/search novelty is
/// checked against the session's persisted structural scope.
pub fn goal_drive_made_progress(
    before: Option<&crate::Goal>,
    after: Option<&crate::Goal>,
    telemetry: &[ProgressEvent],
    changed_files: &[String],
) -> bool {
    if !changed_files.is_empty() || telemetry.iter().any(progress_event_counts_as_plan_drive) {
        return true;
    }
    match (after, before) {
        (Some(after), Some(before)) => after.drive_state_changed_since(before),
        _ => true,
    }
}

/// Turn-local plan progress classifier. Multi-turn frontends must use
/// [`crate::Agent::plan_drive_turn_made_progress`] for cross-turn novelty.
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
    telemetry.iter().any(progress_event_counts_as_plan_drive)
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
    fn drive_progress_counts_novel_read_evidence() {
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
        assert!(plan_drive_made_progress(
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
            goal_drive_made_progress(
                Some(&goal),
                Some(&goal),
                &[],
                &["crates/api/src/lib.rs".into()]
            ),
            "file edits count as goal-drive progress even when the goal state is unchanged"
        );
        assert!(
            goal_drive_made_progress(Some(&goal), Some(&goal), &read, &[]),
            "novel read evidence counts as goal-drive progress"
        );
        assert!(
            !goal_drive_made_progress(Some(&goal), Some(&goal), &[], &[]),
            "an unchanged goal with no file edits is a stall"
        );
        assert_eq!(next_plan_drive_stall(true, false, 3), 4);
        let parked = plan_drive_park_message(Some("1/2 remaining — wire the scheduler"));
        assert!(parked.contains("1/2 remaining — wire the scheduler"));
        assert!(!parked.contains("/retry"));
    }

    #[test]
    fn productive_investigation_can_continue_past_the_stall_threshold() {
        let goal = crate::Goal::new("ship", vec!["investigate".into()]);
        let mut plan_stall = PLAN_DRIVE_STALL_LIMIT.saturating_sub(1);
        let mut goal_stall = GOAL_DRIVE_STALL_LIMIT.saturating_sub(1);

        for turn in 0..8 {
            let evidence = [ProgressEvent {
                kind: "meaningful".into(),
                reason: if turn % 2 == 0 {
                    "new file evidence".into()
                } else {
                    "new targeted search evidence".into()
                },
                signature: Some(format!("turn-{turn}")),
            }];
            plan_stall = next_plan_drive_stall(
                true,
                plan_drive_made_progress(Some("investigate"), Some("investigate"), &evidence, &[]),
                plan_stall,
            );
            goal_stall = if goal_drive_made_progress(Some(&goal), Some(&goal), &evidence, &[]) {
                0
            } else {
                goal_stall.saturating_add(1)
            };
            assert_eq!(plan_stall, 0, "plan drive parked on productive turn {turn}");
            assert_eq!(goal_stall, 0, "goal drive parked on productive turn {turn}");
        }

        assert_eq!(plan_drive_status(true, false, plan_stall), "running");
        assert_eq!(goal_drive_status(true, false, goal_stall), "running");
    }

    #[test]
    fn recovery_instruction_is_step_aware_and_truthful_about_retention() {
        let implementation = plan_recovery_instruction(Some("Implement the parser"));
        assert!(implementation.contains("make the smallest safe mutation"));
        assert!(implementation.contains("dropped prior tool-output contents"));
        assert!(implementation.contains("retaining their fingerprints"));

        let investigation = plan_recovery_instruction(Some("Audit build logs"));
        assert!(investigation.contains("concrete conclusion or validation result"));
        assert!(investigation.contains("dropped prior tool-output contents"));
        assert!(!investigation.contains("make the smallest safe mutation"));
    }

    #[test]
    fn evidence_ledger_does_not_evict_and_recredit_a_finite_cycle() {
        let mut ledger = DriveEvidenceLedger::default();
        let mut telemetry = crate::TurnTelemetry {
            drive_evidence_hashes: (0..300)
                .map(|index| hash_drive_evidence_signature(&format!("read:file-{index}")))
                .collect(),
            ..crate::TurnTelemetry::default()
        };
        assert_eq!(ledger.record_novel(&telemetry).len(), 300);

        telemetry.drive_evidence_hashes = vec![
            hash_drive_evidence_signature("read:file-0"),
            hash_drive_evidence_signature("read:file-299"),
        ];
        assert!(
            ledger.record_novel(&telemetry).is_empty(),
            "crossing the former bounded-ledger size must not recredit old evidence"
        );
    }

    #[test]
    fn unlimited_retry_bookkeeping_still_reaches_the_no_progress_stall_guard() {
        let mut goal = crate::Goal::new("ship", vec!["fix the failing check".into()]);
        let mut stall = 0;

        for attempt in 1..=GOAL_DRIVE_STALL_LIMIT {
            let before = goal.clone();
            assert!(goal.record_failure(
                format!("failed attempt {attempt}"),
                crate::DEFAULT_SUBGOAL_RETRIES,
            ));
            let made_progress = goal_drive_made_progress(Some(&before), Some(&goal), &[], &[]);
            assert!(
                !made_progress,
                "attempt counters and notes are diagnostics, not productive evidence"
            );
            stall = next_plan_drive_stall(true, made_progress, stall);
        }

        assert_eq!(stall, GOAL_DRIVE_STALL_LIMIT);
        assert_eq!(goal_drive_status(true, false, stall), "parked");
        assert_eq!(goal.status, crate::GoalStatus::Active);
        assert_eq!(goal.sub_goals[0].attempts, GOAL_DRIVE_STALL_LIMIT);
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
