//! Read-only **answer steering** and implementation completeness helpers.
//!
//! This module is the Steer-phase half of "review": intent classification,
//! evidence/implementation trackers, preflight call planning, and **answer-repair**
//! metadata for older sessions. Model answers are not rewritten or rejected
//! based on evidence counts, disclaimer phrases, or heading templates.
//!
//! It does **not** own post-mutation **completion review** (`ReviewPolicy` →
//! `ReviewStatus`) or the long-horizon **goal skeptic** — those live in
//! `agent::skeptic` / `verify_outcome` / `goal_turn`.
//!
//! All of this is pure input classification and text generation — none of it
//! touches `Agent` state directly — so it lives outside the main `lib.rs`.

mod constants;
mod goal_kind;
mod implementation;
mod intent;
mod laziness;
mod nudges;
mod preflight;
mod review_repair;
mod routine_git;
mod settlement;
mod stationarity;
mod stop_detector;
mod todo_gate;
mod tool_guardrail;
mod types;

pub(crate) use constants::*;
pub use goal_kind::GoalKind;
pub use implementation::is_destructive_git_restore;
pub(crate) use implementation::*;
pub(crate) use intent::*;
pub(crate) use laziness::{
    ClassifierOutput, LazinessCategory, LazinessConfig, LazinessDecision, NoNudgeReason,
    build_laziness_nudge, claim_evidence_category, evaluate_laziness,
};
pub(crate) use nudges::*;
pub(crate) use preflight::*;
pub(crate) use review_repair::*;
pub use routine_git::{git_command_is_routine, git_command_is_routine_in};
#[cfg(test)]
pub(crate) use settlement::forced_final_answer_is_unusable;
pub(crate) use settlement::no_progress_forced_final_is_unusable;
#[cfg(test)]
pub(crate) use stationarity::MAX_CONSECUTIVE_IDENTICAL_PROBLEMATIC_TOOL_CALLS;
pub(crate) use stationarity::{IdenticalToolCallRun, STATIONARITY_NUDGE};
pub(crate) use stop_detector::{BAIL_CONTINUE_NUDGE, matched_bail_out};
pub(crate) use todo_gate::{
    TodoGateDecision, TodoGateReason, evaluate_todo_gate, todo_gate_input_from_plan,
};
pub(crate) use tool_guardrail::*;
pub(crate) use types::*;
