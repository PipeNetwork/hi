//! `impl Agent` methods split by responsibility. Each submodule re-opens
//! `impl super::Agent` for its group of methods; the struct definition and
//! the orchestration entry points stay in `lib.rs`.

pub(crate) mod audit_goal;
mod compaction_turn;
mod curate_turn;
mod delegate_turn;
mod explore_turn;
mod goal_turn;
mod lifecycle;
mod memory_turn;
mod mutation_recovery_turn;
pub(crate) mod plan_goal;
mod preflight;
pub mod skeptic;
mod tool_selection;
pub(crate) mod trio;
pub(crate) mod turn;

pub(crate) use curate_turn::MAX_AUTO_SKILLS_PER_SESSION;
// Only referenced from tests; the handlers use the consts directly in-module.
#[cfg(test)]
pub(crate) use delegate_turn::MAX_DELEGATE_SUBAGENTS_PER_SESSION;
#[cfg(test)]
pub(crate) use explore_turn::MAX_EXPLORE_SUBAGENTS_PER_SESSION;
