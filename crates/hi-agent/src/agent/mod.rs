//! `impl Agent` methods split by responsibility. Each submodule re-opens
//! `impl super::Agent` for its group of methods; the struct definition and
//! the orchestration entry points stay in `lib.rs`.

pub(crate) mod audit_goal;
mod background_candidate;
mod background_candidate_verification;
mod background_task;
mod child_process_teardown;
mod coding_memory_turn;
mod compaction_job;
mod compaction_turn;
mod curate_turn;
mod delegate_binding;
pub(crate) mod delegate_turn;
mod explore_turn;
mod goal_turn;
mod lifecycle;
mod memory_turn;
mod mutation_recovery_turn;
pub(crate) mod plan_goal;
mod preflight;
mod process_coordination;
mod provider_capability_runtime;
pub mod skeptic;
mod tool_selection;
pub(crate) mod trio;
pub(crate) mod turn;

pub(crate) use compaction_turn::ContextWindowLimits;

// Only referenced from tests; the handlers use the consts directly in-module.
#[cfg(test)]
pub(crate) use delegate_turn::MAX_DELEGATE_SUBAGENTS_PER_TURN;
#[cfg(test)]
pub(crate) use explore_turn::MAX_EXPLORE_SUBAGENTS_PER_TURN;
