//! Read-only **answer steering** and implementation completeness helpers.
//!
//! This module is the Steer-phase half of "review": intent classification,
//! evidence/implementation trackers, preflight call planning, and **answer-repair**
//! modes ([`ReviewRepairMode`] / [`AnswerRepairMode`]) when the model's answer
//! lacks inspected evidence, concrete file citations, or post-edit validation.
//!
//! It does **not** own post-mutation **completion review** (`ReviewPolicy` →
//! `ReviewStatus`) or the long-horizon **goal skeptic** — those live in
//! `agent::skeptic` / `verify_outcome` / `goal_turn`.
//!
//! All of this is pure input classification and text generation — none of it
//! touches `Agent` state directly — so it lives outside the main `lib.rs`.

mod constants;
mod implementation;
mod intent;
mod mutation_recovery;
mod nudges;
mod preflight;
mod review_repair;
mod tool_guardrail;
mod types;

pub(crate) use constants::*;
pub(crate) use implementation::*;
pub(crate) use intent::*;
pub(crate) use mutation_recovery::*;
pub(crate) use nudges::*;
pub(crate) use preflight::*;
pub(crate) use review_repair::*;
pub(crate) use tool_guardrail::*;
pub(crate) use types::*;
