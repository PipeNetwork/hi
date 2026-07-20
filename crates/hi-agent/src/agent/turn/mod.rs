//! The main turn loop and its helpers: `run_turn` (user message → model →
//! tool calls → results → repeat, then workspace repair), `finalize_turn`, and
//! the per-turn steering/tool-selection helpers.
//!
//! Pipeline phases are named in [`phase::TurnPhase`]:
//! `Setup → (Model → Tools → Steer)* → WorkspaceRepair → Settle → Finalize → Done`.
//!
//! Split by responsibility:
//! - [`phase`] — explicit phase enum (WorkspaceRepair vs review Steer repair)
//! - [`progress`] — progress kinds, stall tracking, tool progress labels
//! - [`retry`] — provider/output-cap retry state and **review**-repair budgets
//! - [`helpers`] — telemetry, routing, tool-entry construction
//! - [`setup`] — checkpoints, snapshots, task-context refresh
//! - [`finalize`] — recap call, usage/steer lines, text-tool cleanup
//! - [`verify_run`] — background teardown + [`crate::verify::WorkspaceRepairVerifier`]
//! - [`settlement`] — keep/invalidate a green verify when the tree moves after
//! - [`loop_`] — `run_turn` orchestration

mod fast_feedback;
mod finalize;
mod helpers;
mod loop_;
mod obligation;
pub mod phase;
mod progress;
mod retry;
mod setup;
mod settlement;
mod verify_run;

pub use phase::TurnPhase;

// Re-export nothing publicly; sibling agent modules call Agent methods directly.
